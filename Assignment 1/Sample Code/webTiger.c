#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <fstream>
#include <iomanip>

using namespace std;

int main(int argc, char *argv[]) {
    char javaCall[100], word[100];
    int num_of_links = 0;
    strcat(javaCall, "java getWebPage ");
    strcat(javaCall, argv[1]);
    strcat(javaCall, " > output.html");
    cout << javaCall << endl;
    system(javaCall);
    ifstream webpage("output.html");
    while (!webpage.eof()) {
        webpage >> setw(99) >> word;
        if (strncmp("href", word, 4) == 0) {
            cout << word << endl;
            num_of_links++;
        }
    }
    cout << argv[1] << " had "  << num_of_links << " links\n";
    webpage.close();  return 0;
}