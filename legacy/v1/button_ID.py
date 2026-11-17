import pygame

# Initialize Pygame and the joystick module
pygame.init()
pygame.joystick.init()

# Check if any joystick (racing wheel) is connected
joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    print("No joystick detected!")
else:
    # Initialize the first joystick (racing wheel)
    joystick = pygame.joystick.Joystick(1)
    joystick.init()
    print(f"Connected to joystick: {joystick.get_name()}")

    try:
        while True:
            # Pump the event queue to process input events
            pygame.event.pump()

            # Loop through all buttons and check if they are pressed
            for i in range(joystick.get_numbuttons()):
                if joystick.get_button(i):
                    print(f"Button {i} pressed")

            # Wait for a short period to avoid excessive CPU usage
            pygame.time.wait(100)

    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        # Clean up Pygame
        pygame.quit()
