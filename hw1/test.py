class Sequence:
    def __init__(self, array):
        self.array = array
        
    def __len__(self):
        return len(self.array)
    
    def __iter__(self):
        for item in self.array:
            yield item

class Fibonacci(Sequence):
    def __init__(self, first_value, second_value):
        self.first = first_value
        self.second = second_value
    
    def __call__(self, length):
        self.array = [self.first, self.second]
        for i in range(length-2):
            self.array.append(self.array[-1] + self.array[-2])
        print(self.array)
        
FS = Fibonacci(1 ,2)
FS(length=5)

print("array: ", FS.array, type(FS.array))
print(len(FS))

print([n for n in FS])
# i = iter(FS)
# print(i)
# for n in i:
#     print(n)