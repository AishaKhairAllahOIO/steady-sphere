#include <Servo.h>

Servo servoX;
Servo servoY;

void setup()
{
  Serial.begin(9600);
  servoX.attach(9);
  servoY.attach(10);
}

void loop()
{
  int posX=90;
  int posY=90;

 if(Serial.available())
 {
    String data=Serial.readStringUntil('\n');
    int commaIndex=data.indexOf(',');

    if(commaIndex>0)
    {
      String x=data.substring(0,commaIndex);
      String y=data.substring(commaIndex+1);

      int posX=x.toInt();
      int posY=y.toInt();

      Serial.print("X: ");
      Serial.print(posX);
      Serial.print(" Y: ");
      Serial.println(posY);
    }
    servoX.write(posX);
    servoY.write(posY);
    delay(10);

 }
}
