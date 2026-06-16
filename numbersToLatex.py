import json

filename = input("Dateiname: ")

with open(filename, "r") as f:
    data = json.load(f)

for dataset, d1 in data.items():
    for method, d2 in d1.items():
        # Accuracy
        a1 = round(d2["accuracy"]["1"], 2)
        a5 = round(d2["accuracy"]["5"], 2)

        # Precision mean
        p1 = round(d2["precision"]["1"]["mean"], 2)
        p5 = round(d2["precision"]["5"]["mean"], 2)

        # Precision std
        ps1 = round(d2["precision"]["1"]["std"], 2)
        ps5 = round(d2["precision"]["5"]["std"], 2)

        # Recall mean
        r1 = round(d2["recall"]["1"]["mean"], 2)
        r5 = round(d2["recall"]["5"]["mean"], 2)

        # Recall std
        rs1 = round(d2["recall"]["1"]["std"], 2)
        rs5 = round(d2["recall"]["5"]["std"], 2)

        # F1 mean
        f1 = round(d2["f1"]["1"]["mean"], 2)
        f5 = round(d2["f1"]["5"]["mean"], 2)

        # F1 std
        fs1 = round(d2["f1"]["1"]["std"], 2)
        fs5 = round(d2["f1"]["5"]["std"], 2)

        # Ausgabe
        print(
            method + " " + dataset + ": "
            + f"{a1} & {a5} & "
            + f"{p1} $\\pm$ {ps1} & {p5} $\\pm$ {ps5} & "
            + f"{r1} $\\pm$ {rs1} & {r5} $\\pm$ {rs5} & "
            + f"{r1} $\\pm$ {fs1} & {f5} $\\pm$ {fs5}"
        )
    
    print()