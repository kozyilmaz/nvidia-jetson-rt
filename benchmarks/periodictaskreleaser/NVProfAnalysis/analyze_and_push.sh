#!/bin/bash
# Carries out two tasks: parses the trace_gpu_* files, and updating the online
# timeline data at cs.unc.edu/~otternes/timeline
ruby generate_timeline_json.rb
scp all_data.js otternes@login.cs.unc.edu:/home/otternes/public_html/nvprof/all_data.js
