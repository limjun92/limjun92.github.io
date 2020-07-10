class Email:
    sender = ""

    def send_mail(self, recv, subject, contents):
        print("from : \t" + self.sender)
        print("To : \t" + recv)
        print("subject:" + subject)
        print("contents" + contents)
        print("-" * 20)

e = Email()
e.sender = "01094537706@gmail.com"

recv_list = ['1@gmail.com','2@gmail.com','3@gmail.com']

for recv in recv_list:
    e.send_mail(recv,"비상연락망입니다.", "이 번호로 연락 부탁드립니다.")
'''
from :  01094537706@gmail.com
To :    1@gmail.com
subject:비상연락망입니다.    
--------------------
from :  01094537706@gmail.com
To :    2@gmail.com
subject:비상연락망입니다.    
--------------------
from :  01094537706@gmail.com
To :    3@gmail.com
subject:비상연락망입니다.    
--------------------
'''

