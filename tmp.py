
def sumList(listToSum):
    
    sum = 0


    # czy jest pusta, jak tak zwracam 0 bo wyżej sum = 0 
    if len(listToSum) == 0:
        return sum
    
    # jak nie jest pusta sumuje wszystkie elementy
    for number in listToSum:
        sum += number

    return sum


list = [1, 2 ,3]

sumOfList = sumList(list)

print(sumOfList)