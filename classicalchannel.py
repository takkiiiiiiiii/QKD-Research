import socket
import threading

# サーバーのIPアドレスとポート
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 12001

# クライアントからの接続を待機する関数
def handle_client(client_socket, addr):
    print(f"[*] 新しい接続を受け入れました: {addr[0]}:{addr[1]}")
    try:
        while True:
            # クライアントからのデータを受信
            data = client_socket.recv(4096)
            if not data:
                break

            # 他のクライアントにデータを送信
            for client in list(clients):  # イテレーション中のリスト変更を防ぐためにコピーを作成
                if client != client_socket:
                    try:
                        client.send(data)
                    except OSError:
                        clients.remove(client)
                        client.close()

    except ConnectionResetError:
        print(f"[*] {addr[0]}:{addr[1]} からの接続が切断されました")
    finally:
        if client_socket in clients:
            clients.remove(client_socket)
        client_socket.close()


# サーバーソケットを作成
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_HOST, SERVER_PORT))
server_socket.listen(3)  # Allows up to three connections. (alice, bob and eve)

print("[*] サーバーが起動しました")

clients = []

while len(clients) < 100: 
    client, addr = server_socket.accept()
    clients.append(client)
    print(f"[*] クライアント{len(clients)}が接続しました: {addr[0]}:{addr[1]}")
    threading.Thread(target=handle_client, args=(client, addr)).start()
