from datetime import date 


def generate_list_of_months(from_date:date, to_date:date):
    from_date = date(from_date.year, from_date.month, 1)
    to_date = date(to_date.year, to_date.month, 1)

    current_date = from_date
    dates = []
    while current_date <= to_date:
        dates.append(current_date)
        if current_date.month == 12:
            current_date = date(current_date.year+1, 1,1)
        else:
            current_date = date(current_date.year, current_date.month+1, 1)
    return dates