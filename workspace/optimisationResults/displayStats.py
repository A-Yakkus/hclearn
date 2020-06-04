import matplotlib.pyplot as plt
import pandas as pd
import sys


"""
Args:
Filename
stat column
label column
number of items
"""


def get_n_largest_items_from_data_frame(df, n: int, column: str):
    """
    Creates a sub dataframe of n elements based on column.

    :param df: Original dataframe
    :param n: amount of elements to extract
    :param column: column to base extraction on
    :return: the extracted dataframe and the rest of the dataset
    """
    subset = df.nlargest(n, column)
    others = df[~df.isin(subset)].dropna()
    return subset, others


def prep_data_for_display(df1, column, labels, df2, interlace=False):
    """
    Converts a dataframe to list, with an other item to account
    for items not contained in df1
    :param df1: The dataframe to convert
    :param column: The data column for display
    :param labels: The label column for display
    :param df2: The dataframe containing other data, can be None
    :param interlace: Interlace the data to spread out segments
    :return: The converted data set
    """
    l1 = df1[column].to_list()
    l2 = df1[labels].to_list()
    if df2 is not None:
        l1.append(sum(df2[column]))
        l2.append("Other")
    if not interlace:
        return l1, l2
    else:
        i = l1[:len(l1)//2]
        j = l1[len(l1)-len(i)-1:]
        k = l2[:len(l2)//2]
        li = l2[len(l2)-len(i)-1:]
        a = [None]*(len(l1))
        b = [None]*(len(l2))
        a[1::2] = i
        a[::2] = j
        b[::2] = li
        b[1::2] = k
        return a, b




def get_set_of_items(df, n: int, column: str):
    """
    Automatically get floor(len(df)/2) sets of data
    :param df: The dataframe of all data
    :param n: number of items to extract
    :param column: The column to extract data from
    :return: A list of each set of data and the extra data
    """
    a, b = get_n_largest_items_from_data_frame(df, n, column)
    rs = [[a, b]]
    if len(b) > n:
        s = get_set_of_items(b, n, column)
        rs.extend(s)
        return rs
    return rs


def display_figure(idx: int, num_item: int, cols: str, label: str, data_obj):
    x, lb = prep_data_for_display(data_obj[0], cols, label, data_obj[1])
    fig1, ax1 = plt.subplots(figsize=(4,4))
    plt.rcParams["axes.titlesize"]=16
    ax1.pie(x, startangle=90)
    ax1.legend(lb, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), prop={'size': 16})
    title_a = "Pie chart of the most common functions by {}".format(translator[cols])
    title_b = "\n in format {} from profile statistics,\n with respect to all other functions".format(label)
    ax1.set_title(title_a+title_b)
    plt.savefig(translator[cols]+"learn.png")
"""


fig1, ax1 = plt.subplots(figsize=(4,4))
ax1.pie(sizes, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
ax1.set_title('  Pie Title', loc='left')

plt.tight_layout()
plt.savefig(__file__+"pie.png", bbox_inches='tight')

"""


translator = {
    "tottime":"Total Time",
    "ncalls": "Number of Calls"
}

if __name__ == '__main__':
    #print(plt.rcParams.keys())
    col = "tottime"
    label_col = "filename:lineno(function)"
    num_items = 16
    data = pd.read_csv("learnCPU.csv")
    result = get_set_of_items(data, num_items, col)

    display_figure(1, num_items, col, label_col, result[0])
    #for i in range(len(result)):
    #    display_figure(i+1, num_items, col, label_col, result[i])
    plt.show()


"""
    x, lb = prep_data_for_display(active[0], col, label_col, active[1], 10)
    idx = i+1
    plt.figure(idx)
    plt.pie(x, labels=lb)
    plt.title((i*num_items, idx*num_items, col, label_col))


data = pd.read_csv("CurrentDataStats.csv")
print(type(data))
col="ncalls"
label_col="filename:lineno(function)"
num_items = 20
largestNcalls = data.nlargest(num_items, col)
others = data[~data.isin(largestNcalls)].dropna()
asList = largestNcalls[col].to_list()
lbls = largestNcalls[label_col].to_list()
sum_others = sum(others[col])
asList.insert(num_items//2, sum_others)
lbls.insert(num_items//2, "Other")
plt.figure(1)
plt.pie(asList,labels=lbls)
plt.title("Pie chart of the {0} most common function by {1}\n in format {2} from profile statistics".format(num_items, col, label_col))
plt.show()

"""
#data = np.genfromtxt("CurrentDataStats.csv", skip_header=2, delimiter=',', dtype=np.str)
#graphData = data[:,0].astype(np.float)
#graphData.sort(axis=0)
#print(data[-100:, 0])
#print(graphData.shape)
#plt.figure(1)
#plt.pie(data[0])
#plt.show()
