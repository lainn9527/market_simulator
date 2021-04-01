class ty:
    num = 0
    def __init__(self):
        ty.add_number()
        print(ty.num)

    @classmethod
    def get_count(cls):
        return cls.num

    @classmethod
    def add_number(cls):
        cls.num += 1
        return cls.num

class tyy(ty):
    num = 0
    def __init__(self):
        super().__init__()
        tyy.add_number()
        print('ty', ty.get_count(), 'tyy', self.get_count())

a = ty()
b = ty()
c = tyy()
# d = tyy()