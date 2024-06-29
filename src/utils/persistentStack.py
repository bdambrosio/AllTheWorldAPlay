import pickle
from pathlib import Path


STACK_DIR = Path.home() / '.local/share/AllTheWorld/pStacks/'
STACK_DIR.mkdir(parents=True, exist_ok=True)

class PersistentStack:
    def __init__(self, filename):
        self.filename = filename
        self.stack = []
        self.load()

    def push(self, item):
        self.stack.append(item)
        self.save()

    def pop(self):
        if not self.is_empty():
            item = self.stack.pop()
            self.save()
            return item
        else:
            raise IndexError("Stack is empty. Cannot perform pop operation.")

    def peek(self):
        if not self.is_empty():
            return self.stack[-1]
        else:
            raise IndexError("Stack is empty. Cannot perform peek operation.")

    def is_empty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)

    def clear(self):
        self.stack.clear()
        self.save()

    def save(self):
        with open(STACK_DIR / self.filename, 'wb') as file:
            pickle.dump(self.stack, file)

    def load(self):
        try:
            with open(STACK_DIR / self.filename, 'rb') as file:
                self.stack = pickle.load(file)
        except Exception as e:
            print(f'PersistentStack load error {str(e)}')
            self.stack = []

    def __str__(self):
        return str(self.stack)
    
