# 상속
#```
class Pet:
    def __init__(self, name):
        self.name = name

# Cat이 상위 class
class Cat(Pet):
    def meow(self):
        print(self.name + ' is meowing...')

class Dog(Pet):
    def bark(self):
        print(self.name + ' is barking')

cat1 = Cat('흰돼지')

cat1.meow() # 흰돼지 is meowing...

dog1 = Dog('쎄리')

dog1.bark() # 흰돼지 is barking...
#```

# 예외처리
x = []
try:
    result = x.index(1234)
except ValueError as err:
    print(err)
else:
    print(result)
finally:
    print("끝")