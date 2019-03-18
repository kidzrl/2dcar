import socket
class socket_connect:

    def __init__(self, server = ('192.168.43.204', 10000), client=('192.168.43.2', 10000)):
        self.address = server
        self.readdr = client
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind(self.address)
        self.s.listen(5)
        self.socket_con,_ = self.s.accept()
    def receive(self):
        state_real = self.socket_con.recv(1024)
        return state_real
    def send(self,action):
        self.socket_con.send(action.encode())