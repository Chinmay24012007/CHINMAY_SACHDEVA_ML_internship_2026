#printing the sum of first 10 natural numbers
'''n=0;
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
n1=int(input("enter the first number:"));
n2=int(input("enter the second number:"));
n3=int(input("enter the third number:"));
if n1>n2:
    print("first number is greatest");
elif n2>n3:
    print("second number is the greatest");
else:
    print("third number is the greatest");'''

#student result system

#inputing student details
nos=int(input("enter the number of students you want details of:"));
for i in (1,nos):  
    namee=input("enter the student's name:");
    rno=int(input("enter the student's roll no:"));
    s1=int(input("enter the subject 1 marks:"));
    s2=int(input("enter the subject 2 marks:"));
    s3=int(input("enter the subject 3 marks:"));
    s4=int(input("enter the subject 4 marks:"));
    s5=int(input("enter the subject 5 marks:"));

    #calculating the percentage
    total=s1+s2+s3+s4+s5;
    perc=(total/500)*100;

#displaying the details of the student
for i in (1,nos):
    print("the name of the student is :",namee);
    print("the roll no of the student is:",rno);

#alloting the grades accordinglyy
    if perc==100:
        print("your grade is A+");
    elif perc>=90:
        print("your grade is A");
    elif perc>=80:
        print("your grade is B");
    elif perc>=70:
        print("your grade is C");
    elif perc>=60:
        print("your grade is D");
    elif perc>=50:
        print("you fail,grade is E");


    
