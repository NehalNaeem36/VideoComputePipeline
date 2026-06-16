#include "utils/logger.h"

#include <stdio.h>

int main(void) {
    printf("Running logger tests...\n");
    log_info("info message");
    log_warn("warning message");
    log_error("error message");
    return 0;
}
