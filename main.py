# main.py
from calisthenics.pushup import start_pushups
from calisthenics.squat import start_squats
from calisthenics.pullup import start_pullups
from calisthenics.lunge import start_lunges
from calisthenics.jumping_jacks import start_jumping_jacks

from martial_arts.taekwondo import start_taekwondo
from martial_arts.karate import start_karate

def choose_camera():
    try:
        ci = int(input("Camera index (default 0): ") or "0")
    except:
        ci = 0
    return ci

def main():
    print("""
    Choose mode:
    1. Calisthenics - Push-ups
    2. Calisthenics - Squats
    3. Calisthenics - Pull-ups
    4. Calisthenics - Lunges
    5. Calisthenics - Jumping Jacks
    6. Martial Arts - Taekwondo Kicks
    7. Martial Arts - Karate Punches
    q. Quit
    """)
    choice = input("Enter choice: ").strip().lower()
    cam = choose_camera()

    if choice == '1':
        start_pushups(cam)
    elif choice == '2':
        start_squats(cam)
    elif choice == '3':
        start_pullups(cam)
    elif choice == '4':
        start_lunges(cam)
    elif choice == '5':
        start_jumping_jacks(cam)
    elif choice == '6':
        side = input("Which leg? left/right (default left): ").strip().lower() or 'left'
        start_taekwondo(cam, side=side)
    elif choice == '7':
        side = input("Which hand? left/right (default left): ").strip().lower() or 'left'
        start_karate(cam, side=side)
    elif choice == 'q':
        print("Bye.")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    while True:
        main()
