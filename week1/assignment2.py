#printing the sum of first 10 natural numbers
n=0;
for i in range(1,11):
    n=n+i;
print("the sum of the first 10 natural numbers:",n);

#finding the factorial of a number
num=int(input("enter the number you want to reverse:"));
m=0;
while num>0:
    digit=num%10;
    m=m*10 + digit;
    num=num//10;
print("the reversed number:",m);

#fibonacci series
n = int(input("Enter the number of terms: "))
a = 0
b = 1
print("Fibonacci Series:")
for i in range(n):
    print(a, end=" ");
    c = a + b;
    a = b;
    b = c;

#finding the largest among 3 numbers

