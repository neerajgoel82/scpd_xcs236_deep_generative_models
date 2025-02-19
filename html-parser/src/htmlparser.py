from bs4 import BeautifulSoup


def read_html_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def extract_divs_from_html(content):
    soup = BeautifulSoup(content, 'html.parser')
    divs = soup.find_all('div')
    return divs

def filter_divs_with_data_activity(divs):
    #divs = [div for div in divs if 'activity-item' in div.get('class', [])]
    filtered_divs = [div for div in divs if div.has_attr('data-activityname')]
    return filtered_divs

def extract_activity_names_from_divs(divs):
    activity_names = [div['data-activityname'] for div in divs if 'data-activityname' in div.attrs]
    return activity_names


def create_file(file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(f"\n")

def main():
    print("Hello, World!")
    content = read_html_file("/Users/negoel/xcs236_toc.html")
    divs = extract_divs_from_html(content)
    #filtered_divs = filter_divs_with_data_activity(divs)
    topics = extract_activity_names_from_divs(divs)

    i = 0 
    for topic in topics:
        create_file("video" + str(i) + "_" + topic + ".txt")
        i = i + 1
        print(topic)
    

if __name__ == "__main__":
    main()