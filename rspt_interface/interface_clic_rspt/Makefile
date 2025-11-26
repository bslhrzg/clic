# Use Clang (native macOS compiler) to avoid errors with Apple headers
CC = clang

# 1. Find Python executable (try python3 first, then python)
PYTHON_EXE := $(shell which python3 2>/dev/null || which python 2>/dev/null)

# Check if we found it
ifeq ($(PYTHON_EXE),)
    $(error Python not found! Please run 'mamba install python numpy')
endif

# 2. Get Flags using python-config
# We use the explicit path to ensure we use the config belonging to the python we found
PY_CFLAGS  := $(shell $(PYTHON_EXE)-config --includes)

# 3. Compiler Flags
# -O2: Optimization
# -Wall: Warnings
# -fPIC: Position Independent Code (Critical for linking)
# -Wno-unused-command-line-argument: Clang is picky about some flags
CFLAGS = -O2 -Wall -fPIC -Wno-unused-command-line-argument $(PY_CFLAGS)

TARGET = libimpmod.a
SRCS = src/c_to_python_bridge.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	ar rcs $@ $^
	@echo "--------------------------------------"
	@echo " Library $@ created!"
	@echo " Compiler: $(CC)"
	@echo " Python:   $(PYTHON_EXE)"
	@echo "--------------------------------------"

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean