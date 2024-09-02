# Mnenomic Creator for Falcon (https://falcon-sign.info/)
# by Lars the Penguin

import falcon
import random
from os import urandom

# General word section of https://en.wiktionary.org/wiki/Appendix:Basic_English_word_list
words = ["account", "act", "addition", "adjustment", "advertisement", "agreement", "air", "amount", "amusement", "animal", "answer", "apparatus", "approval", "argument", "art", "attack", "attempt", "attention", "attraction", "authority", "back", "balance", "base", "behaviour", "belief", "birth", "bit", "bite", "blood", "blow", "body", "brass", "bread", "breath", "brother", "building", "burn", "burst", "business", "butter", "canvas", "care", "cause", "chalk", "chance", "change", "cloth", "coal", "colour", "comfort", "committee", "company", "comparison", "competition", "condition", "connection", "control", "cook", "copper", "copy", "cork", "cotton", "cough", "country", "cover", "crack", "credit", "crime", "crush", "cry", "current", "curve", "damage", "danger", "daughter", "day", "death", "debt", "decision", "degree", "design", "desire", "destruction", "detail", "development", "digestion", "direction", "discovery", "discussion", "disease", "disgust", "distance", "distribution", "division", "doubt", "drink", "driving", "dust", "earth", "edge", "education", "effect", "end", "error", "event", "example", "exchange", "existence", "expansion", "experience", "expert", "fact", "fall", "family", "father", "fear", "feeling", "fiction", "field", "fight", "fire", "flame", "flight", "flower", "fold", "food", "force", "form", "friend", "front", "fruit", "glass", "gold", "government", "grain", "grass", "grip", "group", "growth", "guide", "harbour", "harmony", "hate", "hearing", "heat", "help", "history", "hole", "hope", "hour", "humour", "ice", "idea", "impulse", "increase", "industry", "ink", "insect", "instrument", "insurance", "interest", "invention", "iron", "jelly", "join", "journey", "judge", "jump", "kick", "kiss", "knowledge", "land", "language", "laugh", "law", "lead", "learning", "leather", "letter", "level", "lift", "light", "limit", "linen", "liquid", "list", "look", "loss", "love", "machine", "man", "manager", "mark", "market", "mass", "meal", "measure", "meat", "meeting", "memory", "metal", "middle", "milk", "mind", "mine", "minute", "mist", "money", "month", "morning", "mother", "motion", "mountain", "move", "music", "name", "nation", "need", "news", "night", "noise", "note", "number", "observation", "offer", "oil", "operation", "opinion", "order", "organization", "ornament", "owner", "page", "pain", "paint", "paper", "part", "paste", "payment", "peace", "person", "place", "plant", "play", "pleasure", "point", "poison", "polish", "porter", "position", "powder", "power", "price", "print", "process", "produce", "profit", "property", "prose", "protest", "pull", "punishment", "purpose", "push", "quality", "question", "rain", "range", "rate", "ray", "reaction", "reading", "reason", "record", "regret", "relation", "religion", "representative", "request", "respect", "rest", "reward", "rhythm", "rice", "river", "road", "roll", "room", "rub", "rule", "run", "salt", "sand", "scale", "science", "sea", "seat", "secretary", "selection", "self", "sense", "servant", "sex", "shade", "shake", "shame", "shock", "side", "sign", "silk", "silver", "sister", "size", "sky", "sleep", "slip", "slope", "smash", "smell", "smile", "smoke", "sneeze", "snow", "soap", "society", "son", "song", "sort", "sound", "soup", "space", "stage", "start", "statement", "steam", "steel", "step", "stitch", "stone", "stop", "story", "stretch", "structure", "substance", "sugar", "suggestion", "summer", "support", "surprise", "swim", "system", "talk", "taste", "tax", "teaching", "tendency", "test", "theory", "thing", "thought", "thunder", "time", "tin", "top", "touch", "trade", "transport", "trick", "trouble", "turn", "twist", "unit", "use", "value", "verse", "vessel", "view", "voice", "walk", "war", "wash", "waste", "water", "wave", "wax", "way", "weather", "week", "weight", "wind", "wine", "winter", "woman", "wood", "wool", "word", "work", "wound", "writing", "year"]

# Each word has a 4-digit number (0000-9999) added to the end

def createMnemonicAndKey(n=512):
    seed = urandom(32)
    random.seed(seed)
    sk = falcon.SecretKey(n)

    seedAsInt = str(int.from_bytes(seed))
    print(seedAsInt)

    prevI = 0
    mnemonic = []
    for i in range(2, len(str(seedAsInt))+3, 3):
        wordIndex = (int(str(seedAsInt)[prevI:i])//3)
        e = str((int(str(seedAsInt)[prevI:i])%3))
        mnemonic.append(words[wordIndex]+f'{e:01}')
        prevI = i
    
    return sk, mnemonic

def getSecretFromMnemonic(mnemonic, n=512):
    seed = ""
    for i in mnemonic:
        if mnemonic.index(i) == 0 or mnemonic.index(i) == len(mnemonic)-1:
            seed += str(words.index(i[:-1])*3+int(i[-1]))
        else:
            seed += f'{(words.index(i[:-1])*3+int(i[-1])):03}'
    print(seed)
    random.seed(seed)

    sk = falcon.SecretKey(n)
    return sk

if __name__ == '__main__':
    sk, mn = createMnemonicAndKey(512)

    print(mn)

    getSecretFromMnemonic(mn)
