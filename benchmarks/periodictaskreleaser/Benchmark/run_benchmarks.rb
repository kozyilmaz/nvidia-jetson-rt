#!/usr/bin/ruby
# This file will run a the set of benchmarks in parallel. It calls out to the
# command line, and is only tested on bash and Linux.
require 'json'

# The directory of this script.
BASE_DIRECTORY = File.expand_path(File.dirname(__FILE__)) + "/"

# The directory in which to put the results and logs.
OUTPUT_DIRECTORY = BASE_DIRECTORY + "results/"

# The size of inputs, for benchmarks that use it (vector add and matrix mul.)
INPUT_SIZE = 2 ** 22

# The number of seconds for which each scenario is run.
DURATION = 5

# The device on which to run benchmarks, or -1 if no GPU should be explicitly
# specified (e.g. if MPS is being used)
CUDA_DEVICE = 1

# Holds information about a single benchmark program
class Benchmark
  attr_accessor :device, :duration, :name, :include_blocks, :input_size,
    :executable

  # The actual executable for a benchmark will be "benchmark_<name>"
  def initialize(name, include_blocks = false)
    @name = name
    @executable = BASE_DIRECTORY + "benchmark_" + name
    @duration = DURATION
    @input_size = INPUT_SIZE
    @device = CUDA_DEVICE
    @include_blocks = include_blocks
  end

  def to_s
    @name
  end

  # Returns the full command and arguments as a single string.
  def command
    device = ""
    if @device != -1
      device = "--device #{@device.to_s}"
    end
    blocks = ""
    if @include_blocks
      blocks = "--show_blocks"
    end
    "%s --size %d --duration %d --randsleep %s %s" % [@executable, @input_size,
      @duration, blocks, device]
  end
end

# Takes a list of benchmark objects and runs them, spread across different
# processors. This function will create subdirectories in the OUTPUT_DIRECTORY.
# Returns an array of benchmark log file paths, in the order the benchmarks
# were passed to run_scenario. The array of paths returned may contain empty
# strings if log_all is false.
def run_scenario(benchmarks, log_all = true)
  # Outputs = "<total # running>/<specific combination>/<benchmark>/*.csv"
  log_directory = OUTPUT_DIRECTORY + benchmarks.size.to_s + "/"
  names = benchmarks.map{|b| b.name}
  combo = names.sort.each_with_object(Hash.new(0)) {|v, h| h[v] += 1}
  combo = combo.to_a.sort{|a, b| a[0] <=> b[0]}
  combo = combo.map{|v| v[1].to_s + "_" + v[0]}.join("_")
  puts "Running scenario #{combo}"
  log_directory += combo + "/"
  cpu_count = `nproc`.to_i
  cpu_core = 1
  benchmark_count = Hash.new(1)
  pids = []
  to_return = []
  benchmarks.each do |benchmark|
    # Determine the log directory and create it if it doesn't exist.
    log_location = log_directory + benchmark.name + "/"
    `mkdir -p #{log_location}`
    log_location += benchmark_count[benchmark].to_s + ".csv"
    if !log_all && (benchmark_count[benchmark] > 1)
      log_location = "/dev/null"
      to_return << ""
    else
      to_return << log_location
    end
    # Spawn new processes for each benchmark.
    # TODO: Run vmstat before and after each benchmark program.
    puts "  Running #{benchmark.executable} on CPU #{cpu_core.to_s}"
    pids << Process.fork do
      # Set the CPU core for this process and its children.
      `taskset -c -p #{cpu_core.to_s} $$`
      # Execute the command, redirecting to the log, skipping OS buffering
      `stdbuf -oL #{benchmark.command()} > #{log_location}`
    end
    cpu_core = (cpu_core + 1) % cpu_count
    benchmark_count[benchmark] += 1
  end
  pids.each {|pid| Process.wait(pid)}
  to_return
end

# Returns an array of the form [[kernel start, kernel end], ...]. Takes the
# name of a benchmark log file.
def get_kernel_time_list(filename)
  to_return = []
  lines = File.open(filename, "rb") {|f| f.read.split(/\n+/)}
  lines.each do |line|
    kernel_start = 0.0
    kernel_end = 0.0
    # The lines we're interested in shouldn't contain any letters, just numbers
    next if line =~ /[a-zA-Z]/
    values = line.split.map {|v| v.to_f}
    to_return << [values[1], values[2]]
  end
  to_return
end

# Returns an array of the form [[iteration start, iteration end], ...]. Takes
# the name of a benchmark log file.
def get_total_time_list(filename)
  to_return = []
  lines = File.open(filename, "rb") {|f| f.read.split(/\n+/)}
  lines.each do |line|
    kernel_start = 0.0
    kernel_end = 0.0
    # The lines we're interested in shouldn't contain any letters, just numbers
    next if line =~ /[a-zA-Z]/
    values = line.split.map {|v| v.to_f}
    to_return << [values[0], values[3]]
  end
  to_return
end

# Returns an array of the form [[[block start, block end], ...], ...]. There
# are 3 nested arrays. The top level is an array of kernels, the second level
# is an array of blocks, and the bottom level is a 2-element list of start and
# end times of each block.
def get_block_time_list(filename)
  to_return = []
  lines = File.open(filename, "rb") {|f| f.read.split(/\n+/)}
  lines.each do |line|
    next if line !~ /^Block times/
    values = []
    # Block times are given as a space-separated list of start,end pairs. They
    # are in units of seconds * 1e5, so convert them to seconds here for
    # consistency.
    if line =~ /^[^:]+: (.*?)\s*$/
      values = $1.split(" ").map{|r| r.split(",").map{|v| v.to_f / 1e5}}
    end
    to_return << values
  end
  to_return
end

# Takes an array of durations in seconds and converts it to a CDF plot (e.g.
# [[times], [percentages]]). Input times are expected to be in *seconds*, but
# output times will be in milliseconds.
def time_list_to_cdf(durations)
  return [[], []] if durations.size == 0
  durations = durations.sort
  total_count = durations.size.to_f
  current_min = durations[0]
  count = 0.0
  data_list = [durations[0]]
  ratio_list = [0.0]
  durations.each do |point|
    count += 1.0
    if point > current_min
      data_list << point
      ratio_list << count / total_count
      current_min = point
    end
  end
  data_list << durations[-1]
  ratio_list << 1.0
  # Convert times to milliseconds
  data_list.map! {|v| v * 1000.0}
  # Convert ratios to percentages
  ratio_list.map! {|v| v * 100.0}
  [data_list, ratio_list]
end

# Reads output.json from disk and parses it.
def get_output_json()
  return {} if !File.exists?("output.json")
  content = File.open("output.json", "rb") {|f| f.read}
  # In case the file is blank...
  return {} if content.size < 2
  JSON.parse(content)
end

# Takes a hash and saves it to output.json on disk.
def save_output_json(data)
  File.open("output.json", "wb") {|f| f.write(JSON.pretty_generate(data))}
end

mm = Benchmark.new("mm")
va = Benchmark.new("va")
fasthog = Benchmark.new("fasthog")
sd = Benchmark.new("sd")
sd_blocks = Benchmark.new("sd", true)
sd_blocks.duration = 10

# Get kernel and total times for SD depending on competition
4.times do |t|
  count = t + 1
  log_file = run_scenario([sd] * count, false)[0]
  name = "In isolation"
  if t >= 1
    name = "vs. #{t.to_s} instances"
  end
  puts "Reading logs..."
  total_times = get_total_time_list(log_file).map{|v| v[1] - v[0]}
  kernel_times = get_kernel_time_list(log_file).map{|v| v[1] - v[0]}

  # Dump results to disk and run garbage collection after each benchmark to
  # save memory for the next benchmarks.
  puts "Updating output.json..."
  data = get_output_json()
  data["SD total times"] = {} if !data.include?("SD total times")
  data["SD total times"][name] = time_list_to_cdf(total_times)
  # Add the 4 * isolation total time plot
  if name == "In isolation"
    data["SD total times"]["4 * isolation"] = time_list_to_cdf(
        total_times.map {|v| v * 4})
  end
  data["SD kernel times"] = {} if !data.include?("SD kernel times")
  data["SD kernel times"][name] = time_list_to_cdf(kernel_times)
  save_output_json(data)
  puts "Collecting garbage..."
  data = nil
  total_times = nil
  kernel_times = nil
  GC.start()
end

# Get block times for SD depending on competition.
4.times do |t|
  count = t + 1
  log_file = run_scenario([sd_blocks] * count, false)[0]
  name = "In isolation"
  if t >= 1
    name = "vs. #{t.to_s} instances"
  end
  puts "Reading logs..."
  block_times = get_block_time_list(log_file)
  block_times = block_times.map{|k| k.map{|v| v[1] - v[0]}}.flatten

  puts "Updating output.json..."
  data = get_output_json()
  data["SD block times"] = {} if !data.include?("SD block times")
  data["SD block times"][name] = time_list_to_cdf(block_times)
  save_output_json(data)
  puts "Collecting garbage..."
  data = nil
  block_times = nil
  GC.start()
end

