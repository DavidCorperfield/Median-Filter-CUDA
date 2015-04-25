# Path to the cuda installation
CUDA_ROOT = /usr/local/cuda-7.0

# CUDA compiler full path
CC = $(CUDA_ROOT)/bin/nvcc

# Source, Build, BIN, Target directory paths
SRCDIR = src
BUILDDIR = build
BINDIR = bin
TARGET = bin/mf

SRCEXT = cpp

# Include all cpp files to use as sources and feed them to use as objects.
SOURCES = $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJS = $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))

CFLAGS = -std=c++11

# Add the debug flags if you'd like them.
CFLAGS +=

BOOST_ROOT = /usr/include/boost

# Normal libs to always include
LIBS = -lrt -lboost_system -lboost_filesystem

INC = -I $(BOOST_ROOT)

$(TARGET): $(OBJS)
	@mkdir -p $(BINDIR)
	$(CC) $^ -o $(TARGET) $(LIBS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	$(RM) -r $(BUILDDIR) $(TARGET) && $(RM) -r $(BINDIR)

.PHONY: clean

