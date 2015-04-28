# CUDA compiler
CC = nvcc

# Source, Build, BIN, Target directory paths
SRCDIR = src
BUILDDIR = build
BINDIR = bin
TARGET = bin/mf

SRCEXT = cu

# Include all cu files to use as sources and feed them to use as objects.
SOURCES = $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJS = $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))

CFLAGS = -std=c++11

# Add the debug flags if you'd like them.
CFLAGS += -D_DEBUG

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

