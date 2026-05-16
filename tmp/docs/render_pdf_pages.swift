import Foundation
import PDFKit
import AppKit

if CommandLine.arguments.count != 3 {
    fputs("usage: render_pdf_pages.swift input.pdf output_dir\n", stderr)
    exit(2)
}

let input = URL(fileURLWithPath: CommandLine.arguments[1])
let outDir = URL(fileURLWithPath: CommandLine.arguments[2])
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

guard let doc = PDFDocument(url: input) else {
    fputs("failed to open pdf\n", stderr)
    exit(1)
}

let scale: CGFloat = 2.0
for i in 0..<doc.pageCount {
    guard let page = doc.page(at: i) else { continue }
    let box = page.bounds(for: .mediaBox)
    let size = NSSize(width: box.width * scale, height: box.height * scale)
    let image = NSImage(size: size)
    image.lockFocus()
    guard let ctx = NSGraphicsContext.current?.cgContext else {
        image.unlockFocus()
        continue
    }
    NSColor.white.setFill()
    NSRect(origin: .zero, size: size).fill()
    ctx.saveGState()
    ctx.scaleBy(x: scale, y: scale)
    page.draw(with: .mediaBox, to: ctx)
    ctx.restoreGState()
    image.unlockFocus()

    guard let tiff = image.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiff),
          let png = bitmap.representation(using: .png, properties: [:]) else {
        continue
    }
    let path = outDir.appendingPathComponent(String(format: "page-%03d.png", i + 1))
    try png.write(to: path)
}

print("pages=\(doc.pageCount)")
