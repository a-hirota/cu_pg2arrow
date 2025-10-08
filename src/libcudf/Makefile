# Makefile for libpgstrom_cudf.so
#
# Separate library for cuDF integration to avoid header conflicts

CUDF_PREFIX ?= /home/kiban/work/h-zhang/miniconda3/envs/rapids-25.10
CUDF_INCDIR = $(CUDF_PREFIX)/include
CUDF_LIBDIR = $(CUDF_PREFIX)/lib

# Compiler flags
CXX = g++
CUDA_PATH ?= /usr/local/cuda
CXXFLAGS = -std=c++20 -fPIC -O2 -Wall
CXXFLAGS += -I$(CUDF_INCDIR)
CXXFLAGS += -I$(CUDA_PATH)/include
CXXFLAGS += -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

# Linker flags
LDFLAGS = -shared -L$(CUDF_LIBDIR) -lcudf -lrmm
LDFLAGS += -Wl,-rpath,$(CUDF_LIBDIR)

# Target
TARGET = libpgstrom_cudf.so
OBJS = pgstrom_cudf.o

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -shared -o $@ $^ -L$(CUDF_LIBDIR) -Wl,-rpath,$(CUDF_LIBDIR) -lcudf -lrmm -L$(CUDA_PATH)/lib64 -lcuda

pgstrom_cudf.o: pgstrom_cudf.cpp pgstrom_cudf.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

install: $(TARGET)
	@echo "Installing $(TARGET)..."
	@mkdir -p $(DESTDIR)/usr/local/lib
	install -m 755 $(TARGET) $(DESTDIR)/usr/local/lib/
	@echo "Installing header..."
	@mkdir -p $(DESTDIR)/usr/local/include
	install -m 644 pgstrom_cudf.h $(DESTDIR)/usr/local/include/
