CC = nvcc
SRCDIR = src
BUILDDIR = build
BINDIR = bin
TARGET = bin/mf

SRCEXT = cpp
SOURCES = $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJS = $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))

CFLAGS =

# Add the debug flags if you'd like them.
CFLAGS +=

# Normal libs to always include
LIBS =

INC = -I include

$(TARGET): $(OBJS)
	@mkdir -p $(BINDIR)
	$(CC) $^ -o $(TARGET) $(LIBS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	$(RM) -r $(BUILDDIR) $(TARGET) && $(RM) -r $(BINDIR)

.PHONY: clean

