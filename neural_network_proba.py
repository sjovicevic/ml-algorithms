class Proba:

    def __init__(self, funkcija):

        self.funkcija = funkcija


def funkc(a, b):
    return a+b

p = Proba(funkc(2, 3))

print(p.funkcija)