package com.yml.common;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class RGBUtils {

    public static void main(String[]args){
        try {
            BufferedImage bi = ImageIO.read(new File("D:\\workspace\\Utils\\src\\com\\yml\\common\\1.png"));
            for (int i = 0; i < 50; i++) {
                String picString = "";
                for (int j = 0; j < 50; j++) {
                    picString += getRGB(bi, j, i);
                }
                System.out.println(picString);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public static int getRGB(BufferedImage image, int x, int y) {
        int[] rgb = null;
        if (image != null && x < image.getWidth() && y < image.getHeight()) {
            rgb = new int[3];
            int pixel = image.getRGB(x, y);
            rgb[0] = (pixel & 0xff0000) >> 16;
            rgb[1] = (pixel & 0xff00) >> 8;
            rgb[2] = (pixel & 0xff);
        }
        if(null!=rgb&&255==rgb[0]&&255==rgb[1]&&255==rgb[2]){
            return 0;
        }else{
            return 1;
        }
    }
}
