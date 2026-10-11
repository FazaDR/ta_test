
class Params :

    def __init__(self) :
        self.vigi = 0.5
        self.alpha = 4

    def calculate_vigi(self, Layer) :
        temp = self.alpha ** (Layer - 1) * self.vigi
        return temp

if __name__ == "__main__" :
    L = 5 # L is upmost layer
    lvigi = Params()
    total = 0
    for i in range(1, L + 1 ) :
        temp = lvigi.calculate_vigi(i)
        print(f"Vigilance in layer {i} is : {temp}" )
        total = temp + total

    print(f"Psearch is : {total}")



