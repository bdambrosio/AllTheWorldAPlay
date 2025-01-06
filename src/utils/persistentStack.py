import os
from pathlib import Path

class PersistentStack:
    def __init__(self, name):
        # Set up base directory for persistent stacks
        self.base_dir = Path.home() / '.local/share/AllTheWorld/pStacks'
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.name = name
        self.stack_file = self.base_dir / name
        self.stack = []
        
        # Load or create new stack
        if self.stack_file.exists():
            self.load()
        else:
            self.save()

    def load(self):
        """Load stack from text file"""
        with open(self.stack_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.stack = [line.strip() for line in lines if line.strip()]

    def save(self):
        """Save stack in text format"""
        with open(self.stack_file, 'w', encoding='utf-8') as f:
            for item in self.stack:
                f.write(f"{str(item)}\n")

    def push(self, item):
        self.stack.append(item)
        self.save()

    def pop(self):
        if self.stack:
            item = self.stack.pop()
            self.save()
            return item
        return None

    def peek(self):
        if self.stack:
            return self.stack[-1]
        return None

    def clear(self):
        self.stack = []
        self.save()
    
