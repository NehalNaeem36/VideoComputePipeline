/*
 * UI entry module: starts the Dear ImGui desktop wrapper process. It delegates
 * all application state, rendering, and subprocess control to UiApp.
 */
#include "ui_app.h"

#include <exception>
#include <iostream>

int main(int, char **) {
    try {
        vcpui::UiApp app;
        return app.run();
    } catch (const std::exception &ex) {
        std::cerr << "VideoComputePipelineUI failed: " << ex.what() << '\n';
        return 1;
    }
}
