### TEMPERATURE DATA ###


def split_temperature_csv():
    path = "./data/other_data/Temperature_20240828163333.csv"
    station_rows = {}
    header, station = None, None
    # determine which rows to read for each station
    with open(path, "r") as file:
        for line_number, line in enumerate(file.readlines()):
            if line.startswith("#"):
                if line.startswith("#Format: "):
                    header = line.removeprefix("#Format: ")
                elif header is not None and line.startswith("#4530"):
                    if station is not None and station in station_rows:
                        station_rows[station][-1].append(line_number)

                    station = line.removeprefix("#4530").removesuffix("\n")
                    if station not in station_rows:
                        station_rows[station] = []

                    station_rows[station].append([int(line_number) + 1])

    file_length = int(line_number)
    station_rows[station][-1].append(file_length)
    for station, rows in station_rows.items():
        inds = []
        print(rows)
        for r in rows:
            inds += list(np.arange(r[0], r[1] + 1))
        df = pd.read_csv(
            path,
            names=header.split(", "),
            skiprows=list(set(np.arange(file_length)) - set(inds)),
        )
        print(df)
        df.to_csv("./data/temperature/" + station + ".csv")
