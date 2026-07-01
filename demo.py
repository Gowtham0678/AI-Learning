from random import randint

random_num = randint(1, 100)
print("Guess the number between 1 and 100")
tries = 0

while True:
    try:
        guess = int(input("Your guess: "))
    except ValueError:
        print("Invalid input. Please enter an integer.")
        continue

    tries += 1

    if guess == random_num:
        print(f"You win! It took you {tries} guesses.")
        break
    elif guess > random_num:
        print("Too high. Try again.")
    else:
        print("Too low. Try again.")