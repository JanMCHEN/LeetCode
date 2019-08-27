class Trie:
    """
    前缀树，用于单词查找
    """
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._tree = {}
        self.total = 0

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self._tree
        for char in word:
            node = node.setdefault(char, {})
        node['data'] = word
        self.total += 1

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        if self._tree == {}:
            return False
        search_node = self._tree
        for w in word:
            if w in search_node:
                search_node = search_node[w]
            else:
                return False
        if search_node.get('data'):
            return True
        return False

    def start_with(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        start_node = self._tree
        del self._tree
        for pre in prefix:
            if pre in start_node:
                start_node = start_node[pre]
            else:
                return False
        return True



