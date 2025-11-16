import time
import serial


SERIAL_PORT = "COM5"  
BAUD_RATE = 115200
# ====================

def main():
    print(f"Opening serial port {SERIAL_PORT} at {BAUD_RATE}...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # wait for Arduino to reset

    print("Connected.")
    print("Type 1 to blink LED 6 times, 0 to turn LED off, q to quit.")

    try:
        while True:
            user_input = input("Enter (1/0/q): ").strip()

            if user_input.lower() == "q":
                print("Quitting.")
                break
            elif user_input == "1":
                ser.write(b"1\n")   # send '1' + newline
                print("Sent: 1")
            elif user_input == "0":
                ser.write(b"0\n")   # send '0' + newline
                print("Sent: 0")
            else:
                print("Invalid input. Please type 1, 0, or q.")
    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == "__main__":
    main()
