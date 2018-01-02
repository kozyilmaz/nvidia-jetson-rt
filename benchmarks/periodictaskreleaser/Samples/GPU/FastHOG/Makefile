# Based on the project's original makefile, re-organized for simplicity.
.PHONY: all clean

NVCC := nvcc -ccbin g++
NVCCFLAGS := --cudart shared -g --ptxas-options=-v -arch=sm_50
INCLUDES := -I/usr/local/cuda/samples/common/inc -I.

all: fasthog.a

fasthog.a: HOGEngineDevice.o HOGHistogram.o HOGConvolution.o HOGSVMSlider.o \
		HOGEngine.o HOGUtils.o HOGScale.o HOGPadding.o HOGImage.o fastHOG.o
	ar rcsv fasthog.a *.o
	ranlib fasthog.a

fastHOG.o: fastHOG.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

HOGHistogram.o: HOGHistogram.cu HOGHistogram.h
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

HOGConvolution.o: HOGConvolution.cu HOGConvolution.h
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

HOGSVMSlider.o: HOGSVMSlider.cu HOGSVMSlider.h
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

HOGEngine.o: HOGEngine.cu HOGEngine.h
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

HOGEngineDevice.o: HOGEngineDevice.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

HOGUtils.o: HOGUtils.cu HOGUtils.h
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

HOGScale.o: HOGScale.cu HOGScale.h
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

HOGPadding.o: HOGPadding.cu HOGPadding.h
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

HOGImage.o: HOGImage.cu HOGImage.h
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

clean:
	rm -f fasthog.a *.o
