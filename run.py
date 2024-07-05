import schedule
import time
import argparse

def job(args):
    print("I'm working...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("username")
    parser.add_argument("owner")
    parser.add_argument("repo")
    args = parser.parse_args()
    print(args)

    schedule.every(1).minutes.do(job, args=args)

    while True:
        schedule.run_pending()
        time.sleep(1)
