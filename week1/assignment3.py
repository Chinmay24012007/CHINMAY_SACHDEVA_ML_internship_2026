'''#printing first 10 natural numbers
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
reverse(num);'''

#function to count the digits in number
def countt(n):
    c=0;
    for i in range(n):
        c=c+1;
    print("the number of digits in the given number is:",c);
num=int(input("enter the number of which you want to count digits:"));
countt(num);


