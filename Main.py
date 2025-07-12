import cv2
import tkinter as tk
from tkinter import messagebox
import numpy as np
from ultralytics import YOLO
import torch
import threading
import time
import chess
from tkinter import filedialog
from tkinter import simpledialog
import chess.pgn
import chess.engine

#wybór kamery
def choose_camera():
    root_cam = tk.Tk()
    root_cam.withdraw()

    available_cameras = []
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(str(i))
            cap.release()
    if not available_cameras:
        messagebox.showerror("Błąd", "Nie znaleziono żadnej kamery!")
        return None

    camera_index = simpledialog.askinteger(
        "Wybierz kamerę",
        f"Dostępne kamery: {', '.join(available_cameras)}\nWpisz numer kamery:",
        minvalue=0,
        maxvalue=10,
        initialvalue=0 if '0' in available_cameras else int(available_cameras[0])
    )
    root_cam.destroy()

    return camera_index

#ustawianie kamery
camera_index = choose_camera()
if camera_index is None:
    exit()

cap = cv2.VideoCapture(camera_index)

piece_map = {
    '♙': chess.Piece(chess.PAWN, chess.WHITE),
    '♗': chess.Piece(chess.BISHOP, chess.WHITE),
    '♘': chess.Piece(chess.KNIGHT, chess.WHITE),
    '♖': chess.Piece(chess.ROOK, chess.WHITE),
    '♕': chess.Piece(chess.QUEEN, chess.WHITE),
    '♔': chess.Piece(chess.KING, chess.WHITE),

    '♟': chess.Piece(chess.PAWN, chess.BLACK),
    '♝': chess.Piece(chess.BISHOP, chess.BLACK),
    '♞': chess.Piece(chess.KNIGHT, chess.BLACK),
    '♜': chess.Piece(chess.ROOK, chess.BLACK),
    '♛': chess.Piece(chess.QUEEN, chess.BLACK),
    '♚': chess.Piece(chess.KING, chess.BLACK)
}

classNames = ['biala_dama', 'biala_wieza', 'bialy_goniec', 'bialy_kon', 'bialy_krol', 'bialy_pionek', 'czarna_dama', 'czarna_wieza', 'czarny_goniec', 'czarny_kon', 'czarny_krol', 'czarny_pionek', 'reka']

#parametry wideo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

#ładowanie modelu YOLO
model = YOLO("./myChessV9.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)


#parametry planszy
size = 100  #rozmiar pojedynczego pola
detections_history = []  #lista wyników detekcji

#słownik mapujący klasy na znaki ascii figur szachowych
class_to_piece = {
    'biala_dama': '♕', 'biala_wieza': '♖', 'bialy_goniec': '♗', 'bialy_kon': '♘', 'bialy_krol': '♔', 'bialy_pionek': '♙',
    'czarna_dama': '♛', 'czarna_wieza': '♜', 'czarny_goniec': '♝', 'czarny_kon': '♞', 'czarny_krol': '♚', 'czarny_pionek': '♟'
}


#menu do wyboru trybu rozgrywki
def choose_game_mode():
    global play_with_bot

    root_menu = tk.Tk()
    root_menu.title("Wybierz tryb gry")
    root_menu.geometry("300x150")

    label = tk.Label(root_menu, text="Wybierz tryb gry:", font=("Arial", 14))
    label.pack(pady=10)

    def set_mode(mode):
        global play_with_bot
        play_with_bot = (mode == "bot")
        root_menu.destroy()

    pvp_btn = tk.Button(
        root_menu,
        text="Człowiek vs Człowiek",
        command=lambda: set_mode("human"),
        width=20,
        height=2
    )
    pvp_btn.pack(pady=5)

    pvb_btn = tk.Button(
        root_menu,
        text="Człowiek vs Bot",
        command=lambda: set_mode("bot"),
        width=20,
        height=2
    )
    pvb_btn.pack(pady=5)

    root_menu.mainloop()

play_with_bot = None
choose_game_mode()
if play_with_bot is None:
    exit()

#inicjalizacja silnika Stockfish, jeżeli wybierzemy tryb gry z botem
engine = None
if play_with_bot:
    STOCKFISH_PATH = "./stockfish/stockfish-windows-x86-64-avx2.exe"
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        messagebox.showerror("Błąd", str(e))
        engine = None

bot_from_change = None
bot_to_change = None
def make_engine_move():
    global board, current_position, last_position,bot_from_change, bot_to_change

    if engine is None or board is None:
        return
    try:
        result = engine.play(board, chess.engine.Limit(time=0.5))
        move = result.move

        bot_from_row = 7 - (move.from_square // 8)
        bot_from_col = move.from_square % 8
        bot_to_row = 7 - (move.to_square // 8)
        bot_to_col = move.to_square % 8

        bot_from_change = {"row": bot_from_row, "col": bot_from_col}
        bot_to_change = {"row": bot_to_row, "col": bot_to_col}

        board.push(move)

        #aktualizacja planszy do formatu tablicy
        new_array = np.full((8, 8), ' ', dtype=object)
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                row = 7 - (sq // 8)
                col = sq % 8
                for k, v in piece_map.items():
                    if v == piece:
                        new_array[row, col] = k
                        break
        current_position = new_array
        last_position = new_array.copy()
        root.after(2, update_canvas, new_array)
    except Exception as e:
        print(f"Błąd silnika: {e}")


def save_game_to_file():
    global board
    if board is None:
            messagebox.showinfo("Brak partii", "Nie wykryto jeszcze żadnej partii do zapisania!")

    file_path = filedialog.asksaveasfilename(defaultextension=".pgn", filetypes=[("PGN Files", "*.pgn")])
    if not file_path:
        return

    game = chess.pgn.Game.from_board(board)
    with open(file_path, "w", encoding="utf-8") as f:
        print(game, file=f)

    messagebox.showinfo("Zapisano", f"Partia została zapisana do pliku:\n{file_path}")

#funkcja do rysowania szachownicy w canvasie
def draw_board(board, canvas):
    global from_change, to_change, last_move_valid, bot_from_change, bot_to_change

    canvas.delete("all")  # czyszczenie canvas przed rysowaniem

    for row in range(8):
        for col in range(8):
            is_square_light = (row + col) % 2 == 0

            if from_change and to_change and ((row == from_change["row"] and col == from_change["col"]) or (row == to_change["row"] and col == to_change["col"])):
                if last_move_valid:
                    light_color = "#ddffdd" #zielony dla legalnych ruchów
                    dark_color = "#88aa88"
                else:
                    light_color = "#ffdddd" #czerwony dla nielegalnych ruchów
                    dark_color = "#aa8888"
                color = light_color if is_square_light else dark_color
            elif play_with_bot and bot_from_change and bot_to_change and ((row == bot_from_change["row"] and col == bot_from_change["col"]) or (row == bot_to_change["row"] and col == bot_to_change["col"])):
                            light_color = "#ddddff"  #niebieski dla ruchów bota
                            dark_color = "#8888aa"
                            color = light_color if is_square_light else dark_color
            else:
                color = "white" if is_square_light else "gray"

            canvas.create_rectangle(col * size, row * size, (col+1) * size, (row+1) * size, fill=color)

            #rysowanie pionków
            piece = board[row][col]
            if piece != ' ':
                canvas.create_text(col * size + size//2, row * size + size//2, text=piece, font=("Arial", 42), fill="black")

    #rysowanie etykiet A-H i 1-8
    for i in range(8):
        canvas.create_text(i * size + size//2, 8 * size + 10, text=chr(65 + i), font=("Arial", 14))  # Litery A-H
        canvas.create_text(8 * size + 10, i * size + size//2, text=str(8 - i), font=("Arial", 14))  # Cyfry 1-8

#funkcja do uśredniania wyników detekcji z kilku klatek
from collections import Counter

def average_detections(detections_history):
    avg_detections = np.full((8, 8), ' ', dtype=object)
    for i in range(8):
        for j in range(8):
            pieces = [detections_history[k][i][j] for k in range(len(detections_history)) if detections_history[k][i][j] != ' ']
            if pieces:
                counter = Counter(pieces)
                most_common_piece, _ = counter.most_common(1)[0]
                avg_detections[i][j] = most_common_piece
            else:
                avg_detections[i][j] = ' '
    return avg_detections


#funkcja do ładowania klatek
def load_frame():
    global current_frame
    while True:
        success, frame = cap.read()
        if success:
            current_frame = frame
        time.sleep(0.01)  #spowolnienie wczytywania klatek

def array_to_board(position):
    board = chess.Board(None)
    board.turn = chess.WHITE
    #pełne prawa do roszady (białe i czarne, krótka i długa)
    board.castling_rights = (
            chess.BB_H1 | chess.BB_A1 |  # białe: krótka + długa
            chess.BB_H8 | chess.BB_A8  # czarne: krótka + długa
    )

    for row in range(8):
        for col in range(8):
            symbol = position[row][col]
            if symbol != ' ':
                piece = piece_map.get(symbol)
                if piece:
                    board.set_piece_at(chess.square(col, 7 - row), piece)
    return board

from_change = None #pole (row,col) z którego wykonano ruch w notacji(row, col) aby potem w draw_board zaznaczyc to miejsce specjalnym kolorem
to_change = None #pole (row,col) na które wykonano ruch w notacji(row, col) aby potem w draw_board zaznaczyc to miejsce specjalnym kolorem
def detect_move(last_position, current_position):
    global from_change, to_change
    promotion = False
    castling = False  # czy nastąpiła roszada na fizycznej szachownicy, potrzebne, ponieważ jeżeli na szachownicy ruszymy się  np. tylko królem z E1 na C1, to nie powinien być dozwolony tak daleki skok królem, ale obiekt board pozwala na taki ruch bo zakłada, że została wykonana roszada i na obiekcie board też przestawia wieżę, ale fizycznie wieża nie została przecież ruszona
    from_sq = to_sq = None
    figure = None #zmienna figura, aby śledzić, która figura wykonała ruch (potrzebne do głównie do en pasant i promocji pionka)
    changes = 0
    for row in range(8):
        for col in range(8):
            if last_position[row][col] != current_position[row][col]:
                changes += 1
                if current_position[row][col] == ' ':
                    from_sq = chess.square(col, 7 - row)
                    from_change = {"row": row, "col": col}

                    if (last_position[row][col] == '♙' and row == 1) or (last_position[row][col] == '♟' and row == 6):
                        promotion = True
                        print("Promocja pionka!")
                else:
                    to_sq = chess.square(col, 7 - row)
                    to_change = {"row": row, "col": col}
                    figure = current_position[row][col]
    if changes != 2:
        print(f"🛑 Ostrzeżenie: wykryto {changes} zmian (oczekiwane: 2). Możliwy błąd.")
        if changes == 3: #en passant (bicie w przelocie), czyli jeśli są 3 zmiany to poprawić aby from_sq na pewno dotyczył tej samej figury co to_sq, bo w en passant zachodzą 3 zmiany, 2 pola staje się puste, a jedno staje się wypełnione; należy w każdym przypadku from_sq ustawić na pole które stało się puste, ale dotyczące pionka bijącego (tego co teraz jest na to_sq)
            for row in range(8):
                for col in range(8):
                    if last_position[row][col] == figure and current_position[row][col] == " ":
                        from_sq = chess.square(col, 7 - row)
                        from_change = {"row": row, "col": col}
        if changes == 4: #roszady = specyficzne przypadki, czyli jeśli są 4 zmiany na szachownicy upewnić się czy są to roszady i które dokładnie (krókie, długie) (czarne,białe) -> łącznie 4 przypadki
            if ((last_position[0][0] == "♜" and current_position[0][0] == " ") and (last_position[0][3] == " " and current_position[0][3] == "♜")) and ((last_position[0][4] == "♚" and current_position[0][4] == " ") and (last_position[0][2] == " " and current_position[0][2] == "♚")):
                from_sq = chess.square(4, 7) #roszada długa czarne
                to_sq = chess.square(2, 7)
                castling = True
                print("Roszada długa, czarne figury")
            elif ((last_position[0][7] == "♜" and current_position[0][7] == " ") and (last_position[0][5] == " " and current_position[0][5] == "♜")) and ((last_position[0][4] == "♚" and current_position[0][4] == " ") and (last_position[0][6] == " " and current_position[0][6] == "♚")):
                from_sq = chess.square(4, 7) #roszada krótka czarne
                to_sq = chess.square(6, 7)
                castling = True
                print("Roszada krótka, czarne figury")
            elif ((last_position[7][0] == "♖" and current_position[7][0] == " ") and (last_position[7][3] == " " and current_position[7][3] == "♖")) and ((last_position[7][4] == "♔" and current_position[7][4] == " ") and (last_position[7][2] == " " and current_position[7][2] == "♔")):
                from_sq = chess.square(4, 0) #roszada długa białe
                to_sq = chess.square(2, 0)
                castling = True
                print("Roszada długa, białe figury")
            elif ((last_position[7][7] == "♖" and current_position[7][7] == " ") and (last_position[7][5] == " " and current_position[7][5] == "♖")) and ((last_position[7][4] == "♔" and current_position[7][4] == " ") and (last_position[7][6] == " " and current_position[7][6] == "♔")):
                from_sq = chess.square(4, 0) #roszada krótka białe
                to_sq = chess.square(6, 0)
                castling = True
                print("Roszada krótka, białe figury")

    return from_sq, to_sq, castling, promotion, figure


#biblioteka chess do sprawdzania poprawności ruchów
last_position = None
current_position = None
board = None
last_move_valid = True #do oznaczania czy ostatni ruch był dobry czy zły
def checking_if_legal():
    global current_position, last_position, board, last_move_valid, bot_from_change, bot_to_change
    if last_position is None:
        print("Last_postion = Current_position")
        board = array_to_board(current_position)
        print(board)
        last_position = current_position
        return

    from_sq, to_sq, castling_on_camera, promotion, promoted_piece = detect_move(last_position, current_position)
    if from_sq is not None and to_sq is not None:
        print("Kto ma ruch: ", "Białe" if board.turn == chess.WHITE else "Czarne")

        bot_from_change = None
        bot_to_change = None

        if promotion:
            promotion_piece = piece_map[promoted_piece].piece_type
            move = chess.Move(from_sq, to_sq, promotion=promotion_piece)
        else:
            move = chess.Move(from_sq, to_sq)

        isCastlingValid = True
        if board.is_castling(move):
            if castling_on_camera: #samo sprawdzenie czy poprawności roszady na wirutalnej szachownicy nie wystarczy bo szachownica może przyjąć ruch króla z E1 na C1 i wtedy na board przemieści też sama wieżę, mimo że na kamerze(fizycznej szachownicy) na przykład ruch wieży nie zostanie odnotowany -> więc trzeba sprawdzić czy roszada jest wykonana na board oraz czy zgadza sięto ze stanem fizycznej szachownicy
                isCastlingValid = True
            else:
                isCastlingValid = False

        if move in board.legal_moves and isCastlingValid:
            print(f"✅ Legal move: {board.san(move)}")
            board.push(move)
            print(board)
            last_position = current_position
            last_move_valid = True

            if play_with_bot and board.turn == chess.BLACK:
                # ruch czarnych robi silnik
                make_engine_move()

        else:
            print(board)
            print("❌ Illegal move!")
            messagebox.showinfo("Niedozwolony ruch", "Wykonano niedozwolony ruch - cofnij figurę na poprzednie pole!")
            last_move_valid = False

#ręka
hand_present = False
last_hand_time = time.time()
detection_phase = True #false gdy czekamy na usunięcie ręki, true gdy wykonujey detekcje
detection_done = False
#funkcja detekcji na obrazie
def detection_thread():
    global detections_history, current_frame, hand_present, last_hand_time, detection_phase, detection_done, last_position, current_position

    while True:
        if current_frame is not None:
            frame = current_frame.copy()

            # Detekcja obiektów
            results = model(frame, stream=True, verbose=False) #verbose służy do ustawienia logów w konsoli na temat detekcji yolo
            current_detections = [[' ' for _ in range(8)] for _ in range(8)]
            hand_detected = False

            # Parametry planszy
            height, width, _ = frame.shape
            board_size = 1000
            square_size = board_size // 8

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    piece_name = classNames[cls]

                    if piece_name == "reka" and conf > 0.5:
                        hand_detected = True
                        last_hand_time = time.time()

                    elif piece_name != "reka" and detection_phase and conf > 0.5:
                        row = int((center_y - 40) // square_size)
                        col = int((center_x - 50) // square_size)
                        if 0 <= row < 8 and 0 <= col < 8:
                            piece = class_to_piece.get(piece_name, ' ')
                            if 'bialy' in piece_name or 'biala' in piece_name:
                                current_detections[row][col] = piece.upper()
                            else:
                                current_detections[row][col] = piece.lower()

                            color = (0, 255, 0) if 'bialy' in piece_name or 'biala' in piece_name else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, piece_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            #obsługa logiki zmiany stanu
            if hand_detected:
                detection_phase = False  #zatrzymaj detekcje planszy
                detection_done = False
            elif not hand_detected and (time.time() - last_hand_time > 1.5):  # np. 1.5s bez ręki
                detection_phase = True

            if detection_phase and not detection_done:
                detections_history.append(current_detections)

                if len(detections_history) >= 30:
                    averaged_board = average_detections(detections_history)
                    detections_history.clear()
                    detection_done = True
                    detection_phase = False  #czekamy na następną rękę

                    current_position = averaged_board
                    print(averaged_board)
                    checking_if_legal()
                    root.after(1, update_canvas, averaged_board)

            # Rysowanie siatki
            for i in range(9):
                cv2.line(frame, (50 + (i * square_size), 40), (50 + (i * square_size), 40 + board_size), (255, 255, 255), 2)
                cv2.line(frame, (50, 40 + (i * square_size)), (50 + board_size, 40 + (i * square_size)), (255, 255, 255), 2)

            cv2.imshow("Obrazowanie partii szachowej", frame)
            cv2.waitKey(1)


def update_canvas(averaged_board):
    draw_board(averaged_board, canvas)

#tworzenie okna Tkinter
root = tk.Tk()
root.title("Szachownica")

#konfiguracja planszy
canvas = tk.Canvas(root, width=8*size + 30, height=8*size + 30)  # Dodajemy miejsce na numery
save_button = tk.Button(root, text="Zapisz partię", command=save_game_to_file)
save_button.pack(pady=5)
canvas.pack()

#tworzenie wątków
current_frame = None
frame_thread = threading.Thread(target=load_frame)
frame_thread.daemon = True
frame_thread.start()

detection_thread = threading.Thread(target=detection_thread)
detection_thread.daemon = True
detection_thread.start()

#rozpoczęcie głównej pętli GUI
root.mainloop()

#zakończenie wideo
cap.release()
cv2.destroyAllWindows()