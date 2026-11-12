import pygame

class GameController:
    def __init__(self) -> None:
        pygame.init()
        pygame.joystick.init()
        # Select the first connected joystick (e.g., your racing wheel)
        try:
            joystick_count = pygame.joystick.get_count()
            if joystick_count == 0:
                print("No joystick detected!")
                self.joystick = None 
            else:
                self.joystick = pygame.joystick.Joystick(1)  # Use the correct joystick index
                self.joystick.init()
        except Exception as e:
            print(f"Exception in initializing driver data: {e}")

        self.button_id = 8  # The button to hold while speaking
        
    def check_button_press(self):
        """Check for button press on the racing wheel"""
        pygame.event.pump()  # Process event queue
        button_pressed = self.joystick.get_button(self.button_id)

        if button_pressed:
            print("Button pressed. Listening enabled...")
            return True

        elif not button_pressed:
            return False