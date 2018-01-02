files = Dir["trace_gpu_*"]
files.each do |f|
  lines = []
  File.open(f, "rb") {|fl| lines = fl.read.split(/\n+/)}
  lines.delete_if {|line| line !~ /stereoDisparityKernel/}
  lines.map!{|line| line.split(/,/)[1].to_f}
  pid = "<idk>"
  if f =~ /gpu_(\d+)/
    pid = $1
  end
  File.open("kernel_times_" + pid, "wb") {|fl| lines.each {|line| fl.puts line.to_s}}
end
