import chess.pgn
import chess.svg
import chess
import sklearn
import sklearn.ensemble
import sklearn.model_selection
import stockfish
import numpy as np
import joblib
import re
import warnings
#too many warnings from numpy about converting to float
warnings.filterwarnings("ignore", category=DeprecationWarning)

games_list = []
moves_and_evaluations = []
ai_model = None


def main():
   
    #extract_file()
    #read_file()
    #train()
    play_game()

#Game Training Functions
#extracts only high level players from a large list and puts 2000 of them in a new file for reading
def extract_file():
    count = 0
    other_count = 0
    with open("2000_elo_eval.pgn", "w") as pgn_file:
                        pgn_file.write("")
    with open("lichess_db_standard_rated_2014-05.pgn") as file:
        while count < 2000 or other_count < 5000:
            game = chess.pgn.read_game(file)
            game_string = str(game)
            try:
                if int(game.headers["WhiteElo"]) > 2000 and re.search(r'\b(eval)\b', game_string):
                 
                    games_list.append(game)
                    count += 1
                    with open("2000_elo_eval.pgn", "a") as pgn_file:
                        pgn_file.write(str(game)+"\n\n")
                other_count += 1
            except:
                print("bad elo")

            
#reads the file already full of high level players
def read_file():
    #reads in games from filtered games list
    with open("2000_elo_eval.pgn") as file:
        for i in range(200):
            game = chess.pgn.read_game(file)
            
            if game == None:
                break
                
            games_list.append(game)

    #find each evaluation in the moves list and make a list of evals
    for game in games_list:
        moves = []
        evals = []
        #move_pattern = r'\d+\.\s(\S+?)\s*(?:\{[^}]*\[%eval\s*([-+]?\d*\.\d+)\s*]\s*\})?'
        eval_pattern = r'{\s*\[%eval\s+#?([-+]?\d*\.?\d+)\s*\]\s*}'


        evals = re.findall(eval_pattern, str(game.mainline_moves()))

        for move in game.mainline_moves():
            moves.append(str(move))
        
        #put the moves and evaluations together
        if len(moves) != len(evals):
            #print(game.mainline_moves())
            #print(evals)
            #print(moves)
            #print("Length of evals",len(evals))
            #print("Length of moves",len(moves))

            #print("\n")
            #print(moves)
            moves = moves[:len(evals)]
            moves_and_evaluations.append(list(zip(moves, evals)))
        else:
            moves_and_evaluations.append(list(zip(moves, evals)))
        #moves_and_evaluations.append([moves, evals])
        #temp += 1
    #print(moves_and_evaluations)
    #print(temp)

#transforms the list of moves and evalutions into a numpy array so that sklearn can understand it
def fen_to_ml_feature(moves_and_evaluations):
    
    board_array_list = []

    #get each move, push it to the chess board one by one and transform the chessboard into array
    for game in moves_and_evaluations:

        temp_board = chess.Board()
        for pair in game: 
            #push eacn move 
            temp_board.push(chess.Move.from_uci(pair[0]))

            #make a chessboard array
            board_array = np.zeros(64, dtype=np.int8) 
            #if piece is there, fill it with its value
            for square in chess.SQUARES:
                piece = temp_board.piece_at(square)
                if piece is not None:
                    
                    #white capital, black lowercase
                    piece_value = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                                'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6}
                    board_array[square] = piece_value[piece.symbol()]

            #make a list of boards throughout each game
            board_array_list.append(board_array)

    #print(board_array_list)
    #print("Length of board array list (x):",len(board_array_list))           
    return board_array_list


def train():
    #gets the list of numpy boards and puts them into x 
    X_list = fen_to_ml_feature(moves_and_evaluations)
    X = [x for x in X_list]
    #gets the evaluations and puts them into y
    y = []
    for game in moves_and_evaluations:
        for pair in game:
            #print(pair[1])
            y.append(pair[1])

    #train the random forest model
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
    model = sklearn.ensemble.RandomForestClassifier()
    model.fit(X_train, y_train)

    #test accuracy
    accuracy = model.score(X_test, y_test)
    print("Accuracy:", accuracy)
    #save to file
    joblib.dump(model, 'model.joblib')

#transforms the current played game into the numpy array so that the model can use it to predict
def game_board_to_ml_feature(temp_board):
    #makes a chessboard array of 8x8 and puts the value of each piece in its spot
    board_array = np.zeros(64, dtype=np.int8) 
    for square in chess.SQUARES:
        piece = temp_board.piece_at(square)
        if piece is not None:
            
            #white capital, black lowercase
            piece_value = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6}
            board_array[square] = piece_value[piece.symbol()]
    #for some reason it takes a 2d array when using it to predict, so its reshaped
    data_2d = np.reshape(board_array, (1, 64))
    return data_2d


#Game Playing functions
from chessboard import display
def play_game():
    board = chess.Board()
    load_ai()

    while not board.is_game_over():
        #displays game
        game_board = display.start()
    
        display.check_for_quit()
        display.update(board.fen(), game_board)
    
        #gets player moves
        while True:
            player_move = input("Enter move in UCI: ")
            try:
                if chess.Move.from_uci(player_move) in board.legal_moves:
                    board.push(chess.Move.from_uci(player_move))
                    display.update(board.fen(), game_board)
                    break
                else:
                    print("Illegal move.")
            except:
                print("Illegal move.")

        #gets ai moves   
        ai_move = get_ai_move(board)
        print(f"AI move: {ai_move}")
        board.push(ai_move)

    print("Game over. Result: ", board.result())

#loads the ml model from a file
def load_ai():
    global ai_model 
    ai_model = joblib.load("modelv3.joblib", mmap_mode='r')

#minmax function with alpha beta pruning
def min_max(board, depth, alpha, beta, ai_player):
    #end of decision tree, or game over
    if depth <= 0 or board.is_game_over():
        return float(ai_model.predict(game_board_to_ml_feature(board))), None

    #maximizing player
    if ai_player:
        best_eval = float('-inf')
        best_move = None 
        #try each move available to current board and recursively get the evaluation in a depth of 3
        for move in board.legal_moves:
            board.push(move)
            eval, _ = min_max(board, depth - 1, alpha, beta, False)
            #print(type(eval))
            board.pop()  #undo the move
            if eval > best_eval:
                best_eval = eval
                best_move = move
            #alpha beta pruning
            alpha = max(alpha, eval)
            if alpha >= beta:
                break
        return best_eval, best_move
    #minimizing player
    else:
        best_eval = float('inf')
        best_move = None  
        #try each possible move out and send it down recursively
        for move in board.legal_moves:
            board.push(move)
            eval, _ = min_max(board, depth - 1, alpha, beta, True)
            board.pop()  #undo the move
            if eval < best_eval:
                best_eval = eval
                best_move = move
            beta = min(beta, eval)
            #alpha beta pruning
            if alpha >= beta:
                break
        return best_eval, best_move

#main function for handling and getting ai move
def get_ai_move(board):
    #send in a copy of the board so that the original board isnt messed up 
    board_copy = board.copy()
    eval, move = min_max(board_copy, 3, float('-inf'), float('inf'), True)
    return move

if __name__ == "__main__":
    main()