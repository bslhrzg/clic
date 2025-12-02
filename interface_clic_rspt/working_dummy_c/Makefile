# Compiler
CC = gcc

# Compiler flags
# -O2: Optimization level 2
# -Wall: Enable all warnings (good for debugging)
# -fPIC: Position Independent Code (safest for linking with other compiled codes)
CFLAGS = -O2 -Wall -fPIC

# The name of the output library
# Note: It must start with 'lib' and end with '.a' for the -l flag to find it later
TARGET = libimpmod.a

# Source file
SRCS = src/dummy_solver.c

# Object file (derived from source file)
OBJS = $(SRCS:.c=.o)

# Default target (runs when you type 'make')
all: $(TARGET)

# Rule to create the static library archive
$(TARGET): $(OBJS)
	ar rcs $@ $^
	@echo "--------------------------------------"
	@echo " Library $@ created successfully!"
	@echo " Path: $$(pwd)"
	@echo "--------------------------------------"

# Rule to compile .c files into .o files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to clean up files
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean