#printing first 10 natural numbers
def Nnumbers(a,b):
    for i in range(a,b):
        print(i,end=" ");
Nnumbers(1,11);

#sum of first n natrual numbers
def sum(a,b):
    m=0;
    for i in range(a,b):
        m=m+i;
    print("the sum of first",n,"natural numbers is:",m);
n=int(input("\nenter the number of natural numbers you want:"));
sum(1,(n+1));

#function to reverse a  numnber
def reverse(n):
    rev=0;
    while n>0:
        digit=n%10;
        rev=rev*10 + digit;
        n = n//10;
    print("reversed number=",rev);
num=int(input("enter the number you want to reverse:"));
reverse(num);

#function to count the digits in number
def countt(n):
    c=0;
    while n>0:
        c=c+1
        n=n//10
    print("the number of digits in the given number is:",c);
num=int(input("enter the number of which you want to count digits:"));
countt(num);

#function to check a pallindrome number
def pallindrome(n):
    original=n;
    rev=0;
    while n>0:
        digit=n%10;
        rev=rev*10+ digit;
        n=n//10;
    if original==rev:
        print("the number is pallindrome");
    else:
        print("the number is not pallindrome");
num=int(input("enter the number you want to check pallindrome of:"));
pallindrome(num);

#function to generate fibonacci series 
def fibonacci(n):
    a = 0
    b = 1

    print("Fibonacci Series:")
    for i in range(n):
        print(a, end=" ")
        c = a + b
        a = b
        b = c

num = int(input("Enter the number of terms: "))
fibonacci(num)

#calculator using functions
def calculator(n,a,b):
    if n==1:
        print("multiplication of the given numbers:",a*b);
    elif n==2:
        print("division of the given numbers:",a/b);
    elif n==3:
        print("sum of the given numbers:",a+b);
    elif n==4:
        print("difference of the given numbers:",a-b);
print("       MENU:" \
"           1.MULTIPLICATION" \
"           2.DIVISION" \
"           3.ADDITION" \
"           4.SUBTRACTION      ");
num=int(input("enter the number of operation you want to perform:"));
a=int(input("enter the first number:"));
b=int(input("enter the second number:"));
calculator(num,a,b);


