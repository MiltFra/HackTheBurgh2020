import os
if __name__ == "__main__":
  with open("optiver_hacktheburgh/market_data.csv") as f:
    lines = f.readlines()
  new_lines = []
  for l in lines:
    if l.split(',')[1] == "SP-FUTURE":
      new_lines.append(l)
  with open("optiver_hacktheburgh/sp.csv", "w+") as f:
    f.writelines(new_lines)