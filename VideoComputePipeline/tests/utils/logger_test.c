#include "utils/logger.h"

#include <stdio.h>

int main(void) {
    printf("Running logger tests...\n");
    log_info /* module: utils/logger */ ("info message");
    log_warn /* module: utils/logger */ ("warning message");
    log_error /* module: utils/logger */ ("error message");
    return 0;
}
