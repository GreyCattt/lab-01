import tracemalloc
import gc
from array import array
from dataclasses import dataclass

class PostingRegular:
    def __init__(self, doc_id, frequency):
        self.doc_id = doc_id
        self.frequency = frequency

@dataclass(slots=True)
class PostingSlots:
    doc_id: int
    frequency: int

def measure_memory(factory, count=1_000_000):
    gc.collect()
    tracemalloc.start()
    
    if factory == "array":
        data = array('I')
        for i in range(count):
            data.append(i)
            data.append(1)
    else:
        data = [factory(i, 1) for i in range(count)]
        
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)

def main():
    count = 1_000_000
    print(f"Allocating {count} postings...\n")
    
    mem_regular = measure_memory(PostingRegular, count)
    print(f"1. Regular Class: {mem_regular:.2f} MB")
    
    mem_slots = measure_memory(PostingSlots, count)
    print(f"2. Dataclass (slots=True): {mem_slots:.2f} MB")
    
    mem_array = measure_memory("array", count)
    print(f"3. Array('I'): {mem_array:.2f} MB")

if __name__ == "__main__":
    main()