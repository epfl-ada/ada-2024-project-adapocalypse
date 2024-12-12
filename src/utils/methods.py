def process_bechdel_ratings(ratings):
    """
    Process the Bechdel ratings to a more readable format.
    """
    if ratings == 0:
        return "No information"
    elif ratings == 1:
        return "Not passing"
    elif ratings == 2:
        return "Passing"
    elif ratings == 3:
        return "Passing with caveats"
    else:
        return "Invalid rating"