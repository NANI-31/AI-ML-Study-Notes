import re
text = "Hello, how are you?"
m = re.search(r"how", text)
print(m)
if m:
    print(m.start())
else:
    print("Not found")

email = "nani@gmail.com"
check = re.match(r"^[a-zA-Z0-9._]+@[a-zA-Z0-9]+\.[a-zA-Z]{2,}$", email)
if check:
    print("Valid")
else:
    print("Invalid")