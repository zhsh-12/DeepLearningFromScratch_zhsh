#pkl：序列化python对象至字节流/文件，保存进度，再次访问时可快速加载
import pickle, os, time
class GameSave:
    def __init__(self):
        self.saves = {}
    def save_game(self, player_name, level, score, inventory):
        """保存游戏进度"""
        game_data = {
            'level': level,
            'score': score,
            'inventory': inventory,
            'save_time': time.ctime()
        }
        self.saves[player_name] = game_data
        #保存到文件
        with open(f'{player_name}_save.pkl', 'wb') as f:
            pickle.dump(self.saves, f)
        print(f"游戏已保存：{player_name}") #游戏已保存：Tom
    def load_game(self, player_name):
        """加载游戏进度"""
        if os.path.exists(f'{player_name}_save.pkl'):
            with open(f'{player_name}_save.pkl', 'rb') as f:
                return pickle.load(f)
        return None

game = GameSave()
game.save_game('Tom', 5, 2000, ['sword','shield','potion'])
loaded_data = game.load_game('Tom')
print(f"当前游戏进度：{loaded_data}")
#当前游戏进度：{'Tom': {'level': 5, 'score': 2000, 'inventory': ['sword', 'shield', 'potion'], 'save_time': 'Tue Jan 20 10:26:53 2026'}}