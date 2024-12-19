import random

# 오목 보드 크기
BOARD_SIZE = 15
MAX_DEPTH = 2  # 미니맥스 탐색 깊이 제한

# 오목 보드 생성
def create_board():
    return [['.' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

# 보드 출력
def print_board(board):
    print("   " + " ".join([f"{i:2}" for i in range(BOARD_SIZE)]))
    for i, row in enumerate(board):
        print(f"{i:2} " + " ".join([f"{cell:2}" for cell in row]))

# 플레이어의 돌 놓기
def place_stone(board, player, x, y):
    if board[x][y] == '.':
        board[x][y] = player
        return True
    else:
        return False

# 승리 조건 체크
def check_win(board, player):
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if (check_direction(board, player, x, y, 1, 0) or  # 가로
                check_direction(board, player, x, y, 0, 1) or  # 세로
                check_direction(board, player, x, y, 1, 1) or  # 대각선 ↘
                check_direction(board, player, x, y, 1, -1)):  # 대각선 ↙
                return True
    return False

# 특정 방향으로 5개의 돌이 있는지 체크
def check_direction(board, player, x, y, dx, dy):
    count = 0
    for i in range(5):
        nx, ny = x + i * dx, y + i * dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[nx][ny] == player:
            count += 1
        else:
            break
    return count == 5

# 특정 방향으로 3개의 돌이 있는지 체크 (방어용)
def check_three_in_row(board, player, x, y, dx, dy):
    count = 0
    for i in range(3):
        nx, ny = x + i * dx, y + i * dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[nx][ny] == player:
            count += 1
        else:
            break
    return count == 3

# 상대방의 3개의 연속된 돌을 감지하고, 양 끝을 방어하는 함수
def detect_threat_and_block(board, opponent):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 가로, 세로, 대각선 ↘, 대각선 ↙

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x][y] == '.':
                for dx, dy in directions:
                    if check_three_and_block(board, opponent, x, y, dx, dy) or check_four_and_block(board, opponent, x, y, dx, dy):
                        return (x, y)
    return None

# 3개의 돌이 놓인 선의 양 끝이 비어 있는지 확인하고 방어할 위치를 반환하는 함수
def check_three_and_block(board, player, x, y, dx, dy):
    # 좌우 양 끝이 비어 있어야 방어 가능
    count = 0
    start_empty = False
    end_empty = False

    # 3개의 연속된 돌을 확인
    for i in range(-3, 1):
        nx, ny = x + i * dx, y + i * dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            if i == -1 and board[nx][ny] == '.':  # 시작 지점 비어 있음
                start_empty = True
            elif i == 0 and board[nx][ny] == '.':  # 끝 지점 비어 있음
                end_empty = True
            elif board[nx][ny] == player:
                count += 1
            else:
                break
        else:
            break

    # 양 끝이 비어 있고 3개의 연속된 돌이 있다면 true 반환
    return count == 3 and (start_empty or end_empty)

# 4개의 점 중 3개의 돌이 놓여 있고, 양 끝이 비어 있는지 확인하여 방어할 위치 반환
def check_four_and_block(board, player, x, y, dx, dy):
    # 좌우 양 끝이 비어 있어야 방어 가능
    count = 0
    start_empty = False
    end_empty = False

    # 4개의 점 중 3개의 돌을 확인
    for i in range(-3, 1):
        nx, ny = x + i * dx, y + i * dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            if i == -1 and board[nx][ny] == '.':  # 시작 지점 비어 있음
                start_empty = True
            elif i == 3 and board[nx][ny] == '.':  # 끝 지점 비어 있음
                end_empty = True
            elif board[nx][ny] == player:
                count += 1
            else:
                break
        else:
            break

    # 양 끝이 비어 있고 3개의 연속된 돌이 있다면 true 반환
    return count == 3 and start_empty and end_empty

# 보드 상태 평가 함수
def evaluate_board(board, player):
    opponent = 'O' if player == 'X' else 'X'
    player_score = count_score(board, player)
    opponent_score = count_score(board, opponent)
    return player_score - opponent_score

# 간단한 점수 계산 (돌의 연속성에 따라 가중치 부여)
def count_score(board, player):
    score = 0
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            score += check_line_score(board, player, x, y, 1, 0)  # 가로
            score += check_line_score(board, player, x, y, 0, 1)  # 세로
            score += check_line_score(board, player, x, y, 1, 1)  # 대각선 ↘
            score += check_line_score(board, player, x, y, 1, -1)  # 대각선 ↙
    return score

# 연속된 돌의 개수를 점수화 (2개, 3개, 4개 등)
def check_line_score(board, player, x, y, dx, dy):
    score = 0
    count = 0
    for i in range(5):
        nx, ny = x + i * dx, y + i * dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[nx][ny] == player:
            count += 1
        else:
            break
    if count == 2:
        score += 10
    elif count == 3:
        score += 50
    elif count == 4:
        score += 200
    elif count == 5:
        score += 1000
    return score

# 알파-베타 가지치기 적용 미니맥스 알고리즘
def minimax(board, depth, is_maximizing, player, alpha, beta):
    if depth == 0 or check_win(board, 'X') or check_win(board, 'O'):
        return evaluate_board(board, player), None

    opponent = 'O' if player == 'X' else 'X'
    best_move = None

    if is_maximizing:
        max_eval = float('-inf')
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x][y] == '.':
                    board[x][y] = player
                    evaluation, _ = minimax(board, depth - 1, False, player, alpha, beta)
                    board[x][y] = '.'
                    if evaluation > max_eval:
                        max_eval = evaluation
                        best_move = (x, y)
                    alpha = max(alpha, evaluation)
                    if beta <= alpha:
                        break  # 베타 컷오프
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x][y] == '.':
                    board[x][y] = opponent
                    evaluation, _ = minimax(board, depth - 1, True, player, alpha, beta)
                    board[x][y] = '.'
                    if evaluation < min_eval:
                        min_eval = evaluation
                        best_move = (x, y)
                    beta = min(beta, evaluation)
                    if beta <= alpha:
                        break  # 알파 컷오프
        return min_eval, best_move

# AI의 3개의 연속된 돌을 감지하고, 4개의 돌을 만들 수 있는 경우 우선적으로 공격하는 함수
def detect_ai_opportunity_and_complete(board, player):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 가로, 세로, 대각선 ↘, 대각선 ↙

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x][y] == '.':
                for dx, dy in directions:
                    if check_three_and_complete(board, player, x, y, dx, dy):
                        return (x, y)
    return None

# 3개의 연속된 AI의 돌이 있고, 양 끝이 비어 있으면 4개의 돌을 완성할 수 있는지 확인
def check_three_and_complete(board, player, x, y, dx, dy):
    count = 0
    start_empty = False
    end_empty = False

    # 3개의 연속된 AI의 돌을 확인
    for i in range(-3, 1):
        nx, ny = x + i * dx, y + i * dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            if i == -1 and board[nx][ny] == '.':  # 시작 지점 비어 있음
                start_empty = True
            elif i == 0 and board[nx][ny] == '.':  # 끝 지점 비어 있음
                end_empty = True
            elif board[nx][ny] == player:
                count += 1
            else:
                break
        else:
            break

    # AI가 3개의 연속된 돌을 가지고 있고, 양 끝이 비어 있으면 4개의 돌을 만들 수 있음
    return count == 3 and (start_empty or end_empty)

# AI가 연속된 4개의 돌을 완성하여 게임을 끝낼 수 있는지 확인하는 함수
def detect_ai_win_opportunity(board, player):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 가로, 세로, 대각선 ↘, 대각선 ↙

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x][y] == '.':
                for dx, dy in directions:
                    if check_four_and_complete(board, player, x, y, dx, dy):
                        return (x, y)
    return None

# 4개의 연속된 AI의 돌이 있고 5개의 돌을 만들 수 있는지 확인하는 함수
def check_four_and_win(board, player, x, y, dx, dy):
    count = 0

    # 4개의 연속된 AI의 돌을 확인
    for i in range(-4, 1):
        nx, ny = x + i * dx, y + i * dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            if board[nx][ny] == player:
                count += 1
            elif board[nx][ny] != '.':
                break
        else:
            break

    # 4개의 연속된 돌이 있고 빈 자리가 있으면 5개로 완성할 수 있음
    return count == 4

# 상대방이 연속된 3개의 돌을 놓고 양 끝이 비어 있는지 확인하여 방어할 위치를 반환하는 함수
def detect_threat_and_block_extended(board, opponent):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 가로, 세로, 대각선 ↘, 대각선 ↙

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x][y] == '.':
                for dx, dy in directions:
                    if check_three_and_block_extended(board, opponent, x, y, dx, dy):
                        return (x, y)
    return None

# 3개의 연속된 상대방의 돌을 확인하고, 양 끝이 비어 있는지 확인하여 방어
def check_three_and_block_extended(board, opponent, x, y, dx, dy):
    count = 0
    start_empty = False
    end_empty = False

    # 3개의 연속된 상대방의 돌을 확인
    for i in range(-3, 1):
        nx, ny = x + i * dx, y + i * dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            if i == -1 and board[nx][ny] == '.':  # 시작 지점 비어 있음
                start_empty = True
            elif i == 0 and board[nx][ny] == '.':  # 끝 지점 비어 있음
                end_empty = True
            elif board[nx][ny] == opponent:
                count += 1
            else:
                break
        else:
            break

    # 양 끝이 비어 있고 상대방의 3개의 연속된 돌이 있는 경우
    return count == 3 and start_empty and end_empty

# 4개의 연속된 AI의 돌이 있고, 한 곳만 비어 있을 경우 해당 위치에 돌을 놓아 게임을 끝냄
def check_four_and_complete(board, player, x, y, dx, dy):
    count = 0
    empty_pos = None

    # 4개의 연속된 AI의 돌을 확인
    for i in range(-4, 1):
        nx, ny = x + i * dx, y + i * dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            if board[nx][ny] == player:
                count += 1
            elif board[nx][ny] == '.':
                if empty_pos is None:
                    empty_pos = (nx, ny)
                else:
                    break  # 두 개 이상의 빈 자리가 있으면 패턴이 맞지 않음
        else:
            break

    # 4개의 연속된 돌이 있고 한 곳이 비어 있으면 그곳에 돌을 놓을 수 있음
    return count == 4 and empty_pos == (x, y)

# AI가 가장자리에 두지 않도록 하는 조건을 추가한 함수
def valid_move(x, y):
    # 가장자리를 제외한 부분에서만 유효한 수를 두도록 함
    return 1 <= x < BOARD_SIZE - 1 and 1 <= y < BOARD_SIZE - 1

# AI의 돌 놓기 (우선적으로 승리, 그다음 방어, 그다음 공격)
def ai_move(board, player, turn):
    opponent = 'O' if player == 'X' else 'X'

    if turn == 0:
        # 첫 수는 중앙에 두기
        x, y = BOARD_SIZE // 2, BOARD_SIZE // 2
    else:
        # AI가 5개의 돌을 완성할 수 있는지 확인
        win_opportunity = detect_ai_win_opportunity(board, player)
        if win_opportunity:
            x, y = win_opportunity
        else:
            # 상대방이 4개의 점 중 3개의 돌을 놓고 양 끝이 비어 있는지 확인하고 방어
            extended_threat = detect_threat_and_block_extended(board, opponent)
            if extended_threat:
                x, y = extended_threat
            else:
                # AI가 3개의 연속된 돌을 가지고 있는지 확인하고 4개로 만들 수 있는지 확인
                opportunity = detect_ai_opportunity_and_complete(board, player)
                if opportunity and valid_move(opportunity[0], opportunity[1]):
                    x, y = opportunity
                else:
                    # 상대방이 3개의 연속된 돌을 놓았는지 확인하고 방어
                    threat = detect_threat_and_block(board, opponent)
                    if threat:
                        x, y = threat
                    else:
                        # 미니맥스를 통해 최선의 수를 찾음 (가장자리를 피함)
                        _, move = minimax(board, MAX_DEPTH, True, player, float('-inf'), float('inf'))
                        if move and valid_move(move[0], move[1]):
                            x, y = move
                        else:
                            # 유효한 수가 없으면 가장자리에서도 수를 둠
                            x, y = move

    place_stone(board, player, x, y)
    print(f"\nAI가 ({x}, {y}) 위치에 돌을 놓았습니다.")


# 게임 메인 루프
def main():
    board = create_board()
    ai_player = 'X'  # AI가 먼저 시작
    human_player = 'O'
    turn = 0

    print("오목 게임을 시작합니다!")
    print_board(board)

    while True:
        
        if turn % 2 == 0:
            # AI의 차례
            print(f"\n{ai_player}의 차례입니다.")
            ai_move(board, ai_player, turn)
            print_board(board)
            turn += 1

            if check_win(board, ai_player):
                print(f"\n{ai_player}가 승리했습니다! 게임 종료.")
                break
        else:
            # 인간 플레이어의 차례
            print(f"\n{human_player}의 차례입니다.")
            try:
                x, y = map(int, input("돌을 놓을 위치를 입력하세요. (100,100 입력 시 강제 종료) (x, y): ").split())
                if x == 100 and y == 100 :
                    print('게임이 종료되었습니다.')
                    break
            except ValueError:
                print("잘못된 입력입니다. 다시 입력하세요.")
                continue

            if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
                print("잘못된 범위입니다. 다시 입력하세요.")
                continue

            if place_stone(board, human_player, x, y):
                print_board(board)
                turn += 1
            else:
                print("이미 돌이 놓여 있습니다. 다른 위치를 선택하세요.")
                continue

            if check_win(board, human_player):
                print(f"\n{human_player}가 승리했습니다! 게임 종료.")
                break

if __name__ == "__main__":
    main()
