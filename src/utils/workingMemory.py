import json
import pickle
#
## Working Memory for plan execution: a simple key-addressable dict for now
## need to integrate this with OwlCot workingMemory.

### now used in react!

#

class WorkingMemory:
    def __init__(self, name):
        self._wm = {}
        self.filename=name+'_wm'
        
    def save(self, filename):
        # note we need to update wm when awm changes! tbd
        with open(self.filename+'.pkl', 'wb') as f:
          pickle.dump(self._wm, f)
   
    def load(self, filename):
       try:
          with open(self.filename+'.pkl', 'rb') as f:
             self._wm = pickle.load(f)
             print(f'loaded {len(self._wm.keys())} items from {filename}.pkl')
       except Exception as e:
           print(f'load failed, creating {str(e)}')
           self._wm = {}
      
    def has(self, name):
        return name in self._wm
        
    def get(self, name):
        if self.has(name):
            return self._wm[name]
        else:
            return None
        
    def select(self, query):
        """ exhaustive for now, assuming enough context size """
        content = [self._wm[key] for key in self._wm.keys()]
        return '\n'.join(content)

    def show(self):
        print(json.dumps(self._wm, indent=2))

    def assign(self, name, item, type=str, notes=''):
        #print(f'assign {name}, {type}, {str(item)[:32]}')
        if type not in [str, int, dict, list, 'action', 'plan']:
            print (f'unknown type for wm item {type}')
            raise BaseException(f'bad wm type {type}')
        elif ((type(item) in [str, int] and type(item) != type)
              or (type(item) is dict and type not in [dict, 'plan', 'action'])
              or (type in [dict, 'action'] and type(item) is not dict)
              or (type is list and type(item) is not list)):
            print (f'type mismatch, declared: {type}, actual: {type(item)}')
            raise BaseException(f'bad wm type {type}')

        # add entry to Working memory
        #print(f'{self.filename} creating new wm item with name {name}')
        self._wm[name] = {"name":name, "item":item, "type":str(type), "notes":notes}
        return self._wm[name]

    def delete (self, name):
        if self.has(name):
            del self._wm[name]
            return True
        return False

    def clear (self):
        self._wm={}
        
