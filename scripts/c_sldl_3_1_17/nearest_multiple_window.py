
#%%----------------------------------------------------------------

def nearest_multiple(number, multiple):
    """
    Deep Learning pooling layers reduce the time-series by a factor of "multiple".
    This method ensures that the time-series windows is a multiple of the factor, and that 
    avoids runnings errors. 

    """

    nearest = round(number / multiple) * multiple  # nearest_multiple

    print("Nearest multiple of", multiple, "to", number, "is:", nearest)

    return nearest

# test windowing
window_size = int(8*1000)  # ms
step = int(2.25*1000) #ms
overlap = True

window_size = nearest_multiple(window_size, multiple=16)
step = nearest_multiple(step, multiple=16)  #ensures perc. of overlap is exact

print("Window size (sec): ", window_size/1000)
print("step (sec): ", step/1000)
print("overlap: ", overlap)
if overlap:
    print("perc. of overlap: ", (window_size-step)*100/window_size)

