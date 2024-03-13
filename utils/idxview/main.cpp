#include <gempyre.h>
#include <gempyre_utils.h>
#include <gempyre_graphics.h>
#include "resources.h"
#include "idxreader.h"
#include <sstream>
#include <iostream>

std::string info(const Neuropia::IdxReaderBase& reader) {
    return std::to_string(reader.dimensions()) + " " + std::to_string(reader.elementSize());
}


std::string_view right(std::string_view str, size_t p) {
    return str.substr(str.length()- p);
}

void app_main(Gempyre::Ui& ui, int argc, char** argv) {
    const auto& [params, options] = GempyreUtils::parse_args(argc, argv, {});
    if(params.size() < 3) {
        ui.alert("Bad arguments - expectd IMAGES.idx LABELS.idx INDEX");
        return;
    }

    Neuropia::IdxReader<unsigned char> images(params[0]);
    if(!images.ok() || images.dimensions() != 3 || images.type() != Neuropia::IdxReaderBase::Type::Byte) {
        ui.alert("Bad images: " + params[0] + " ( " + info(images) + ")");
        return; 
    }

    Neuropia::IdxReader<unsigned char> labels(params[1]);
    if(!labels.ok() || labels.dimensions() != 1 || images.type() != Neuropia::IdxReaderBase::Type::Byte) {
        ui.alert("Bad labels: " + params[1]  + " ( " + info(labels) + ")" );
        return; 
    }

    if(images.size() != labels.size()) {
        ui.alert("Images and labels are expected to have same size " + std::to_string(images.size()) + " != " + std::to_string(labels.size()));
        return;
    }

    const auto index = GempyreUtils::parse<unsigned>(params[2]);
    if(!index || *index >= images.size()) {
        ui.alert("Bad index " + params[2]);
        return;
    }

    const auto imageSize = images.size(1) * images.size(2);
    const auto image = images.readAt(*index, imageSize); 
    Gempyre::Bitmap bmp(images.size(1), images.size(2));
    unsigned pos = 0;
    for(auto y = 0; y < bmp.height(); ++y) {
        for(auto x = 0; x < bmp.width(); ++x) {
            const auto pix = image[pos++];
            bmp.set_pixel(x, y, Gempyre::Color::rgb(pix, pix, pix)); // grayscale
        }
    }
    const auto png = bmp.png_image();    // make PNG so it can be resized (28x28 is so tiny!)

    Gempyre::CanvasElement canvas(ui, "canvas");
    auto rect = canvas.rect();
    constexpr auto KEY_URL = "/label_number";
    [[maybe_unused]] auto ok = ui.add_data(KEY_URL, png);
    gempyre_utils_assert(ok);

    canvas.add_image(KEY_URL, [canvas, rect = std::move(rect)](const auto& id) {
        canvas.paint_image(id, Gempyre::Element::Rect{0, 0, rect->width, rect->height});
    });

    const auto label = labels.readAt(*index);
    std::stringstream ss;
    ss <<  "\"" << label << "\", 0x" << right(GempyreUtils::to_hex(static_cast<int>(label)), 2) << ", " + std::to_string(static_cast<unsigned>(label));  
    Gempyre::Element(ui, "content").set_html(ss.str());
 }

int main(int argc, char** argv) {
    Gempyre::Ui ui({ {"/gui.html", Guihtml} }, "gui.html");
    ui.on_open([&](){app_main(ui, argc, argv);});
    ui.run();
    return 0;
}